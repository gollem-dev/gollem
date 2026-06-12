package openai

import (
	"github.com/gollem-dev/gollem"
	gollemschema "github.com/gollem-dev/gollem/internal/schema"
	"github.com/sashabaranov/go-openai"
)

// convertTool converts gollem.Tool to openai.Tool
func convertTool(tool gollem.Tool) openai.Tool {
	parameters := make(map[string]interface{})
	properties := make(map[string]interface{})
	spec := tool.Spec()

	for name, param := range spec.Parameters {
		properties[name] = convertParameterToSchema(param)
	}

	if len(properties) > 0 {
		parameters["type"] = "object"
		parameters["properties"] = properties
		if required := gollemschema.CollectRequiredFields(spec.Parameters); len(required) > 0 {
			parameters["required"] = required
		}
	}

	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        spec.Name,
			Description: spec.Description,
			Parameters:  parameters,
		},
	}
}

// convertParameterToSchema converts gollem.Parameter to OpenAI schema
func convertParameterToSchema(param *gollem.Parameter) map[string]interface{} {
	schema := map[string]interface{}{
		"type":        getOpenAIType(param.Type),
		"description": param.Description,
		"title":       param.Title,
	}

	if len(param.Enum) > 0 {
		schema["enum"] = param.Enum
	}

	if param.Properties != nil {
		properties := make(map[string]interface{})
		for name, prop := range param.Properties {
			properties[name] = convertParameterToSchema(prop)
		}
		schema["properties"] = properties
		if required := gollemschema.CollectRequiredFields(param.Properties); len(required) > 0 {
			schema["required"] = required
		}
	}

	if param.Items != nil {
		schema["items"] = convertParameterToSchema(param.Items)
	}

	// Add number constraints
	if param.Type == gollem.TypeNumber || param.Type == gollem.TypeInteger {
		if param.Minimum != nil {
			schema["minimum"] = *param.Minimum
		}
		if param.Maximum != nil {
			schema["maximum"] = *param.Maximum
		}
	}

	// Add string constraints
	if param.Type == gollem.TypeString {
		if param.MinLength != nil {
			schema["minLength"] = *param.MinLength
		}
		if param.MaxLength != nil {
			schema["maxLength"] = *param.MaxLength
		}
		if param.Pattern != "" {
			schema["pattern"] = param.Pattern
		}
	}

	// Add array constraints
	if param.Type == gollem.TypeArray {
		if param.MinItems != nil {
			schema["minItems"] = *param.MinItems
		}
		if param.MaxItems != nil {
			schema["maxItems"] = *param.MaxItems
		}
	}

	// Add default value
	if param.Default != nil {
		schema["default"] = param.Default
	}

	return schema
}

func getOpenAIType(paramType gollem.ParameterType) string {
	switch paramType {
	case gollem.TypeString:
		return "string"
	case gollem.TypeNumber:
		return "number"
	case gollem.TypeInteger:
		return "integer"
	case gollem.TypeBoolean:
		return "boolean"
	case gollem.TypeArray:
		return "array"
	case gollem.TypeObject:
		return "object"
	default:
		return "string"
	}
}
