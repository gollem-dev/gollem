package ollama_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/ollama"
	"github.com/m-mizutani/gt"
)

type noParamTool struct{}

func (noParamTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{Name: "now", Description: "Current time"}
}

func (noParamTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return nil, nil
}

type nestedTool struct{}

func (nestedTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name: "order",
		Parameters: map[string]*gollem.Parameter{
			"items": {
				Type: gollem.TypeArray,
				Items: &gollem.Parameter{
					Type: gollem.TypeObject,
					Properties: map[string]*gollem.Parameter{
						"id":   {Type: gollem.TypeString, Required: true},
						"size": {Type: gollem.TypeString, Enum: []string{"S", "M", "L"}},
					},
				},
			},
		},
	}
}

func (nestedTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return nil, nil
}

func decodeJSON(t *testing.T, data []byte) map[string]any {
	t.Helper()
	var out map[string]any
	gt.NoError(t, json.Unmarshal(data, &out)).Required()
	return out
}

func TestToolConversion(t *testing.T) {
	t.Run("tool without parameters", func(t *testing.T) {
		data, err := ollama.ToolJSON(noParamTool{})
		gt.NoError(t, err).Required()
		gt.Equal(t, decodeJSON(t, data), map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "now",
				"description": "Current time",
				"parameters":  map[string]any{"type": "object", "properties": map[string]any{}},
			},
		})
	})

	t.Run("nested object, array and enum", func(t *testing.T) {
		data, err := ollama.ToolJSON(nestedTool{})
		gt.NoError(t, err).Required()
		params := decodeJSON(t, data)["function"].(map[string]any)["parameters"].(map[string]any)
		gt.Equal(t, params["properties"], any(map[string]any{
			"items": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type":                 "object",
					"additionalProperties": false,
					"required":             []any{"id"},
					"properties": map[string]any{
						"id":   map[string]any{"type": "string"},
						"size": map[string]any{"type": "string", "enum": []any{"S", "M", "L"}},
					},
				},
			},
		}))
		_, hasRequired := params["required"]
		gt.False(t, hasRequired)
	})
}

func TestResponseFormat(t *testing.T) {
	t.Run("text without schema", func(t *testing.T) {
		format, err := ollama.ResponseFormat(gollem.ContentTypeText, nil, nil)
		gt.NoError(t, err).Required()
		gt.Value(t, format).Nil()
	})

	t.Run("schema with a map", func(t *testing.T) {
		param := &gollem.Parameter{
			Type: gollem.TypeObject,
			Properties: map[string]*gollem.Parameter{
				"labels": {
					Type:                 gollem.TypeObject,
					AdditionalProperties: &gollem.Parameter{Type: gollem.TypeString},
				},
			},
		}
		format, err := ollama.ResponseFormat(gollem.ContentTypeJSON, param, nil)
		gt.NoError(t, err).Required()
		labels := decodeJSON(t, format)["properties"].(map[string]any)["labels"].(map[string]any)
		gt.Equal(t, labels["additionalProperties"], any(map[string]any{"type": "string"}))
	})

	t.Run("schema with a union", func(t *testing.T) {
		for range 20 {
			format, err := ollama.ResponseFormat(gollem.ContentTypeJSON, newBlocksParameter(), nil)
			gt.NoError(t, err).Required()
			gt.Equal(t, blocksJSONSchema, string(format))
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

// blocksPropertySchema is the JSON Schema of the "blocks" property of
// newBlocksParameter. The union has no "type"; each element has its own.
const blocksPropertySchema = `{"items":{"anyOf":[` +
	`{"additionalProperties":false,"properties":{"kind":{"enum":["paragraph"],"type":"string"},"text":{"type":"string"}},"required":["kind","text"],"type":"object"},` +
	`{"additionalProperties":false,"properties":{"kind":{"enum":["callout"],"type":"string"},"text":{"type":"string"},"tone":{"enum":["info","warning"],"type":"string"}},"required":["kind","text"],"type":"object"}` +
	`],"description":"a block"},"type":"array"}`

// blocksJSONSchema is the JSON Schema of newBlocksParameter.
const blocksJSONSchema = `{"additionalProperties":false,"properties":{"blocks":` + blocksPropertySchema +
	`},"required":["blocks"],"type":"object"}`

type blocksTool struct{}

func (blocksTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{Name: "write_blocks", Parameters: newBlocksParameter().Properties}
}

func (blocksTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return nil, nil
}

func TestToolConversionAnyOf(t *testing.T) {
	for range 20 {
		data, err := ollama.ToolJSON(blocksTool{})
		gt.NoError(t, err).Required()
		gt.Equal(t, `{"type":"function","function":{"name":"write_blocks","parameters":{"properties":{"blocks":`+
			blocksPropertySchema+`},"required":["blocks"],"type":"object"}}}`, string(data))
	}
}
