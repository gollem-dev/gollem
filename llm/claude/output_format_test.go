package claude_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/m-mizutani/gt"
)

func TestOutputSchema(t *testing.T) {
	type testCase struct {
		param    *gollem.Parameter
		expected string
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			actual, err := json.Marshal(claude.OutputSchema(tc.param))
			gt.NoError(t, err)
			assertJSONEqual(t, tc.expected, actual)
		}
	}

	intPtr := func(v int) *int { return &v }
	floatPtr := func(v float64) *float64 { return &v }

	t.Run("nested objects get additionalProperties false", runTest(testCase{
		param: &gollem.Parameter{
			Type: gollem.TypeObject,
			Properties: map[string]*gollem.Parameter{
				"inner": {
					Type:       gollem.TypeObject,
					Required:   true,
					Properties: map[string]*gollem.Parameter{"flag": {Type: gollem.TypeBoolean}},
				},
				"items": {
					Type: gollem.TypeArray,
					Items: &gollem.Parameter{
						Type:       gollem.TypeObject,
						Properties: map[string]*gollem.Parameter{"id": {Type: gollem.TypeString, Required: true}},
					},
				},
			},
		},
		expected: `{
			"type": "object",
			"additionalProperties": false,
			"required": ["inner"],
			"properties": {
				"inner": {
					"type": "object",
					"additionalProperties": false,
					"properties": {"flag": {"type": "boolean"}}
				},
				"items": {
					"type": "array",
					"items": {
						"type": "object",
						"additionalProperties": false,
						"required": ["id"],
						"properties": {"id": {"type": "string"}}
					}
				}
			}
		}`,
	}))

	t.Run("supported keywords are kept", runTest(testCase{
		param: &gollem.Parameter{
			Type:        gollem.TypeArray,
			Description: "Colors",
			MinItems:    intPtr(1),
			Items: &gollem.Parameter{
				Type:    gollem.TypeString,
				Enum:    []string{"red", "blue"},
				Default: "red",
			},
		},
		expected: `{
			"type": "array",
			"description": "Colors",
			"minItems": 1,
			"items": {"type": "string", "enum": ["red", "blue"], "default": "red"}
		}`,
	}))

	t.Run("unsupported constraints move to the description", runTest(testCase{
		param: &gollem.Parameter{
			Type:        gollem.TypeString,
			Description: "Code",
			MinLength:   intPtr(2),
			MaxLength:   intPtr(8),
			Pattern:     "^[A-Z]+$",
		},
		expected: `{
			"type": "string",
			"description": "Code\n\n{minLength: 2, maxLength: 8, pattern: ^[A-Z]+$}"
		}`,
	}))

	t.Run("numeric bounds move to the description without one", runTest(testCase{
		param: &gollem.Parameter{
			Type:    gollem.TypeNumber,
			Minimum: floatPtr(0.5),
			Maximum: floatPtr(10),
		},
		expected: `{"type": "number", "description": "{minimum: 0.5, maximum: 10}"}`,
	}))

	t.Run("minItems above 1 and maxItems move to the description", runTest(testCase{
		param: &gollem.Parameter{
			Type:     gollem.TypeArray,
			MinItems: intPtr(2),
			MaxItems: intPtr(5),
			Items:    &gollem.Parameter{Type: gollem.TypeInteger},
		},
		expected: `{
			"type": "array",
			"description": "{minItems: 2, maxItems: 5}",
			"items": {"type": "integer"}
		}`,
	}))
}

func TestOutputSchemaAnyOf(t *testing.T) {
	t.Run("union elements are strict objects and the union has no type", func(t *testing.T) {
		const expected = `{"additionalProperties":false,"properties":{"blocks":{"items":{"anyOf":[` +
			`{"additionalProperties":false,"properties":{"kind":{"enum":["paragraph"],"type":"string"},"text":{"type":"string"}},"required":["kind","text"],"type":"object"},` +
			`{"additionalProperties":false,"properties":{"kind":{"enum":["callout"],"type":"string"},"text":{"type":"string"},"tone":{"enum":["info","warning"],"type":"string"}},"required":["kind","text"],"type":"object"}` +
			`],"description":"a block"},"type":"array"}},"required":["blocks"],"type":"object"}`

		param := newBlocksParameter()
		for range 20 {
			out, err := json.Marshal(claude.OutputSchema(param))
			gt.NoError(t, err)
			gt.Equal(t, expected, string(out))
		}
	})

	t.Run("rejected constraints are restated in each element's description", func(t *testing.T) {
		maxLength := 10
		param := &gollem.Parameter{AnyOf: []*gollem.Parameter{
			{Type: gollem.TypeString, MaxLength: &maxLength},
			{Type: gollem.TypeInteger},
		}}
		out, err := json.Marshal(claude.OutputSchema(param))
		gt.NoError(t, err)
		gt.Equal(t, `{"anyOf":[{"description":"{maxLength: 10}","type":"string"},{"type":"integer"}]}`, string(out))
	})
}

// newUnionProperties returns an object with n properties, each a union of a
// string and an integer.
func newUnionProperties(n int) *gollem.Parameter {
	props := make(map[string]*gollem.Parameter, n)
	for i := range n {
		props[fmt.Sprintf("p%02d", i)] = &gollem.Parameter{AnyOf: []*gollem.Parameter{
			{Type: gollem.TypeString},
			{Type: gollem.TypeInteger},
		}}
	}
	return &gollem.Parameter{Type: gollem.TypeObject, Properties: props}
}

func TestOutputFormatUnionLimit(t *testing.T) {
	t.Run("16 union parameters pass", func(t *testing.T) {
		format, err := claude.OutputFormat(newUnionProperties(16))
		gt.NoError(t, err)
		gt.NotNil(t, format.Schema)
	})

	t.Run("17 union parameters are rejected", func(t *testing.T) {
		_, err := claude.OutputFormat(newUnionProperties(17))
		gt.True(t, errors.Is(err, gollem.ErrUnsupportedSchema))
		gt.S(t, err.Error()).Contains("schema has 17 parameters with anyOf, more than the 16")
	})
}
