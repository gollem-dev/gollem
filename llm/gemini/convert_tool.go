package gemini

import (
	"github.com/gollem-dev/gollem"
	gollemschema "github.com/gollem-dev/gollem/internal/schema"
	"google.golang.org/genai"
)

// convertTool converts gollem.Tool to Gemini tool
func convertTool(tool gollem.Tool) *genai.FunctionDeclaration {
	spec := tool.Spec()

	// Collect required fields from parameters
	required := gollemschema.CollectRequiredFields(spec.Parameters)
	if required == nil {
		required = []string{}
	}

	parameters := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: make(map[string]*genai.Schema),
		Required:   required,
	}

	for name, param := range spec.Parameters {
		parameters.Properties[name] = convertParameterToSchema(param)
	}

	return &genai.FunctionDeclaration{
		Name:        spec.Name,
		Description: spec.Description,
		Parameters:  parameters,
	}
}

// convertParameterToSchema converts gollem.Parameter to Gemini schema
func convertParameterToSchema(param *gollem.Parameter) *genai.Schema {
	schema := &genai.Schema{
		Type:        getGeminiType(param.Type),
		Description: param.Description,
		Title:       param.Title,
	}

	if len(param.Enum) > 0 {
		schema.Enum = param.Enum
	}

	if param.Properties != nil {
		schema.Properties = make(map[string]*genai.Schema)
		for name, prop := range param.Properties {
			schema.Properties[name] = convertParameterToSchema(prop)
		}
		if required := gollemschema.CollectRequiredFields(param.Properties); len(required) > 0 {
			schema.Required = required
		} else {
			schema.Required = []string{}
		}
	}

	if param.Items != nil {
		schema.Items = convertParameterToSchema(param.Items)
	}

	// Add number constraints
	if param.Type == gollem.TypeNumber || param.Type == gollem.TypeInteger {
		if param.Minimum != nil {
			minVal := *param.Minimum
			schema.Minimum = &minVal
		}
		if param.Maximum != nil {
			maxVal := *param.Maximum
			schema.Maximum = &maxVal
		}
	}

	// Add string constraints
	if param.Type == gollem.TypeString {
		if param.MinLength != nil {
			minLen := int64(*param.MinLength)
			schema.MinLength = &minLen
		}
		if param.MaxLength != nil {
			maxLen := int64(*param.MaxLength)
			schema.MaxLength = &maxLen
		}
		if param.Pattern != "" {
			schema.Pattern = param.Pattern
		}
	}

	// Add array constraints
	if param.Type == gollem.TypeArray {
		if param.MinItems != nil {
			minItems := int64(*param.MinItems)
			schema.MinItems = &minItems
		}
		if param.MaxItems != nil {
			maxItems := int64(*param.MaxItems)
			schema.MaxItems = &maxItems
		}
	}

	// No default value in Gemini

	return schema
}

func getGeminiType(paramType gollem.ParameterType) genai.Type {
	switch paramType {
	case gollem.TypeString:
		return genai.TypeString
	case gollem.TypeNumber:
		return genai.TypeNumber
	case gollem.TypeInteger:
		return genai.TypeInteger
	case gollem.TypeBoolean:
		return genai.TypeBoolean
	case gollem.TypeArray:
		return genai.TypeArray
	case gollem.TypeObject:
		return genai.TypeObject
	default:
		return genai.TypeString
	}
}
