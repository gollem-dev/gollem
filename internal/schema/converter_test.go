package schema_test

import (
	"encoding/json"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/internal/schema"
	"github.com/m-mizutani/gt"
)

// newMultiRequiredParameter returns an object parameter whose properties contain
// several required fields at both the top level and a nested level. Map
// iteration order is randomized per range statement, so a schema built from it
// is a reliable way to observe non-deterministic ordering.
func newMultiRequiredParameter() *gollem.Parameter {
	return &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"zulu":    {Type: gollem.TypeString, Required: true},
			"alpha":   {Type: gollem.TypeString, Required: true},
			"mike":    {Type: gollem.TypeString, Required: true},
			"bravo":   {Type: gollem.TypeString, Required: true},
			"charlie": {Type: gollem.TypeString},
			"nested": {
				Type:     gollem.TypeObject,
				Required: true,
				Properties: map[string]*gollem.Parameter{
					"yankee": {Type: gollem.TypeString, Required: true},
					"delta":  {Type: gollem.TypeString, Required: true},
					"oscar":  {Type: gollem.TypeString, Required: true},
				},
			},
		},
	}
}

func TestCollectRequiredFields(t *testing.T) {
	t.Run("returns required names in ascending order", func(t *testing.T) {
		param := newMultiRequiredParameter()
		gt.Equal(t, []string{"alpha", "bravo", "mike", "nested", "zulu"},
			schema.CollectRequiredFields(param.Properties))
	})

	t.Run("returns the same order on every call", func(t *testing.T) {
		param := newMultiRequiredParameter()
		first := schema.CollectRequiredFields(param.Properties)
		for i := 0; i < 100; i++ {
			gt.Equal(t, first, schema.CollectRequiredFields(param.Properties))
		}
	})

	t.Run("returns nil when no property is required", func(t *testing.T) {
		properties := map[string]*gollem.Parameter{
			"alpha": {Type: gollem.TypeString},
		}
		gt.Nil(t, schema.CollectRequiredFields(properties))
	})
}

func TestConvertParameterToJSONSchemaIsByteStable(t *testing.T) {
	param := newMultiRequiredParameter()

	first, err := json.Marshal(schema.ConvertParameterToJSONSchema(param))
	gt.NoError(t, err)

	for i := 0; i < 100; i++ {
		actual, err := json.Marshal(schema.ConvertParameterToJSONSchema(param))
		gt.NoError(t, err)
		gt.Equal(t, string(first), string(actual))
	}
}

func TestConvertParameterToJSONStringIsByteStable(t *testing.T) {
	param := newMultiRequiredParameter()

	first, err := schema.ConvertParameterToJSONString(param)
	gt.NoError(t, err)

	for i := 0; i < 100; i++ {
		actual, err := schema.ConvertParameterToJSONString(param)
		gt.NoError(t, err)
		gt.Equal(t, first, actual)
	}
}

// The result is embedded verbatim into the Claude system prompt, so the model must read the
// characters the schema author wrote, not encoding/json's HTML escapes for them.
func TestConvertParameterToJSONStringDoesNotEscapeHTML(t *testing.T) {
	param := &gollem.Parameter{
		Type:        gollem.TypeObject,
		Description: `set when a<b && b>c`,
		Properties: map[string]*gollem.Parameter{
			"expr": {
				Type:        gollem.TypeString,
				Description: `an expression such as "x > 1 & y < 2"`,
				Required:    true,
			},
		},
	}

	out, err := schema.ConvertParameterToJSONString(param)
	gt.NoError(t, err)

	// With HTML escaping on, these substrings would appear as unicode escapes instead.
	gt.S(t, out).Contains(`set when a<b && b>c`)
	gt.S(t, out).Contains(`x > 1 & y < 2`)

	// It must still be valid JSON.
	var decoded map[string]any
	gt.NoError(t, json.Unmarshal([]byte(out), &decoded))
}

func newMapParameter(value gollem.ParameterType) *gollem.Parameter {
	return &gollem.Parameter{
		Type:                 gollem.TypeObject,
		AdditionalProperties: &gollem.Parameter{Type: value},
	}
}

func TestFindAdditionalProperties(t *testing.T) {
	type testCase struct {
		param *gollem.Parameter
		path  string
		found bool
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			path, found := schema.FindAdditionalProperties(tc.param)
			gt.Equal(t, tc.found, found)
			gt.Equal(t, tc.path, path)
		}
	}

	t.Run("no map", runTest(testCase{
		param: newMultiRequiredParameter(),
	}))

	t.Run("root map", runTest(testCase{
		param: newMapParameter(gollem.TypeString),
		path:  "(root)",
		found: true,
	}))

	t.Run("map property", runTest(testCase{
		param: &gollem.Parameter{
			Type: gollem.TypeObject,
			Properties: map[string]*gollem.Parameter{
				"name":   {Type: gollem.TypeString},
				"labels": newMapParameter(gollem.TypeString),
			},
		},
		path:  "labels",
		found: true,
	}))

	t.Run("map in array items", runTest(testCase{
		param: &gollem.Parameter{
			Type: gollem.TypeObject,
			Properties: map[string]*gollem.Parameter{
				"items": {
					Type: gollem.TypeArray,
					Items: &gollem.Parameter{
						Type: gollem.TypeObject,
						Properties: map[string]*gollem.Parameter{
							"attrs": newMapParameter(gollem.TypeInteger),
						},
					},
				},
			},
		},
		path:  "items[].attrs",
		found: true,
	}))

	t.Run("nested property", runTest(testCase{
		param: &gollem.Parameter{
			Type: gollem.TypeObject,
			Properties: map[string]*gollem.Parameter{
				"a": {
					Type: gollem.TypeObject,
					Properties: map[string]*gollem.Parameter{
						"b": newMapParameter(gollem.TypeBoolean),
					},
				},
			},
		},
		path:  "a.b",
		found: true,
	}))

	t.Run("first map in name order", func(t *testing.T) {
		param := &gollem.Parameter{
			Type: gollem.TypeObject,
			Properties: map[string]*gollem.Parameter{
				"zeta":  newMapParameter(gollem.TypeString),
				"alpha": newMapParameter(gollem.TypeString),
			},
		}
		for range 20 {
			path, found := schema.FindAdditionalProperties(param)
			gt.True(t, found)
			gt.Equal(t, "alpha", path)
		}
	})
}

// newBlocksParameter returns an object whose "blocks" array holds a union of
// two object kinds told apart by a single-value "kind" enum.
func newBlocksParameter() *gollem.Parameter {
	return &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"blocks": {
				Type:     gollem.TypeArray,
				Required: true,
				Items: &gollem.Parameter{
					Description: "a block",
					AnyOf: []*gollem.Parameter{
						{
							Type: gollem.TypeObject,
							Properties: map[string]*gollem.Parameter{
								"kind": {Type: gollem.TypeString, Enum: []string{"paragraph"}, Required: true},
								"text": {Type: gollem.TypeString, Required: true},
							},
						},
						{
							Type: gollem.TypeObject,
							Properties: map[string]*gollem.Parameter{
								"kind": {Type: gollem.TypeString, Enum: []string{"callout"}, Required: true},
								"text": {Type: gollem.TypeString, Required: true},
								"tone": {Type: gollem.TypeString, Enum: []string{"info", "warning"}},
							},
						},
					},
				},
			},
		},
	}
}

func TestConvertParameterToJSONSchemaAnyOf(t *testing.T) {
	const expected = `{"additionalProperties":false,"properties":{"blocks":{"items":{"anyOf":[` +
		`{"additionalProperties":false,"properties":{"kind":{"enum":["paragraph"],"type":"string"},"text":{"type":"string"}},"required":["kind","text"],"type":"object"},` +
		`{"additionalProperties":false,"properties":{"kind":{"enum":["callout"],"type":"string"},"text":{"type":"string"},"tone":{"enum":["info","warning"],"type":"string"}},"required":["kind","text"],"type":"object"}` +
		`],"description":"a block"},"type":"array"}},"required":["blocks"],"type":"object"}`

	param := newBlocksParameter()
	for range 20 {
		out, err := json.Marshal(schema.ConvertParameterToJSONSchema(param))
		gt.NoError(t, err)
		gt.Equal(t, expected, string(out))
	}
}

func TestConvertParameterToJSONStringAnyOfAtRoot(t *testing.T) {
	param := newBlocksParameter().Properties["blocks"].Items
	out, err := schema.ConvertParameterToJSONString(param)
	gt.NoError(t, err)

	var decoded map[string]any
	gt.NoError(t, json.Unmarshal([]byte(out), &decoded))
	_, hasType := decoded["type"]
	gt.False(t, hasType)
	gt.A(t, decoded["anyOf"].([]any)).Length(2)
	gt.Equal(t, "a block", decoded["description"])
}

func TestFindAdditionalPropertiesInAnyOf(t *testing.T) {
	t.Run("map in a union element", func(t *testing.T) {
		param := newBlocksParameter()
		param.Properties["blocks"].Items.AnyOf[1].Properties["attrs"] = newMapParameter(gollem.TypeString)
		path, found := schema.FindAdditionalProperties(param)
		gt.True(t, found)
		gt.Equal(t, "blocks[].anyOf[1].attrs", path)
	})

	t.Run("map as a union element at the root", func(t *testing.T) {
		param := &gollem.Parameter{AnyOf: []*gollem.Parameter{
			{Type: gollem.TypeString},
			newMapParameter(gollem.TypeString),
		}}
		path, found := schema.FindAdditionalProperties(param)
		gt.True(t, found)
		gt.Equal(t, "(root).anyOf[1]", path)
	})

	t.Run("union without a map", func(t *testing.T) {
		_, found := schema.FindAdditionalProperties(newBlocksParameter())
		gt.False(t, found)
	})
}

func TestCountAnyOf(t *testing.T) {
	t.Run("no union", func(t *testing.T) {
		gt.Equal(t, 0, schema.CountAnyOf(newMultiRequiredParameter()))
	})

	t.Run("union inside array items", func(t *testing.T) {
		gt.Equal(t, 1, schema.CountAnyOf(newBlocksParameter()))
	})

	t.Run("unions at the root, in properties, in map values and inside elements", func(t *testing.T) {
		union := func() *gollem.Parameter {
			return &gollem.Parameter{AnyOf: []*gollem.Parameter{
				{Type: gollem.TypeString},
				{Type: gollem.TypeInteger},
			}}
		}
		param := &gollem.Parameter{AnyOf: []*gollem.Parameter{
			{
				Type: gollem.TypeObject,
				Properties: map[string]*gollem.Parameter{
					"a": union(),
					"b": {Type: gollem.TypeObject, AdditionalProperties: union()},
				},
			},
			union(),
		}}
		gt.Equal(t, 4, schema.CountAnyOf(param))
	})
}

func TestConvertParameterToJSONSchemaMap(t *testing.T) {
	t.Run("map object has a value schema and no properties", func(t *testing.T) {
		out, err := json.Marshal(schema.ConvertParameterToJSONSchema(newMapParameter(gollem.TypeString)))
		gt.NoError(t, err)
		gt.Equal(t, `{"additionalProperties":{"type":"string"},"type":"object"}`, string(out))
	})

	t.Run("map nested in an object", func(t *testing.T) {
		param := &gollem.Parameter{
			Type: gollem.TypeObject,
			Properties: map[string]*gollem.Parameter{
				"labels": newMapParameter(gollem.TypeInteger),
			},
		}
		out, err := json.Marshal(schema.ConvertParameterToJSONSchema(param))
		gt.NoError(t, err)
		gt.Equal(t, `{"additionalProperties":false,"properties":{"labels":{"additionalProperties":{"type":"integer"},"type":"object"}},"type":"object"}`, string(out))
	})

	t.Run("object with properties and a value schema", func(t *testing.T) {
		param := &gollem.Parameter{
			Type: gollem.TypeObject,
			Properties: map[string]*gollem.Parameter{
				"name": {Type: gollem.TypeString},
			},
			AdditionalProperties: &gollem.Parameter{Type: gollem.TypeInteger},
		}
		out, err := json.Marshal(schema.ConvertParameterToJSONSchema(param))
		gt.NoError(t, err)
		gt.Equal(t, `{"additionalProperties":{"type":"integer"},"properties":{"name":{"type":"string"}},"type":"object"}`, string(out))
	})
}
