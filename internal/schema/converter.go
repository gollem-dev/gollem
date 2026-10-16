package schema

import (
	"fmt"
	"maps"
	"slices"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/internal/jsonutil"
	"github.com/m-mizutani/goerr/v2"
)

// CollectRequiredFields returns a list of required property names in ascending
// name order. The order must not depend on Go map iteration: the result is
// emitted verbatim as the JSON Schema "required" array, and Anthropic matches
// the prompt cache on an exact byte prefix that starts with the tool
// definitions, so a reordered array invalidates every cache breakpoint.
func CollectRequiredFields(properties map[string]*gollem.Parameter) []string {
	var required []string
	for name, prop := range properties {
		if prop.Required {
			required = append(required, name)
		}
	}
	slices.Sort(required)
	return required
}

// FindAdditionalProperties returns the path of the first object that has
// AdditionalProperties (a map), visiting properties in ascending name order so
// that the result does not depend on map iteration. The path joins property
// names with "." and marks array items with "[]" and union elements with
// ".anyOf[i]", e.g. "items[].attrs" or "blocks[].anyOf[1].attrs"; the root
// itself is reported as "(root)".
//
// Providers use it to decide how to send a schema: Claude structured outputs
// and OpenAI strict mode reject a schema that contains a map, and Gemini sends
// such a schema as JSON Schema because genai.Schema cannot express a map.
func FindAdditionalProperties(param *gollem.Parameter) (path string, found bool) {
	return findAdditionalProperties(param, "")
}

func findAdditionalProperties(param *gollem.Parameter, path string) (string, bool) {
	if param == nil {
		return "", false
	}
	if param.Type == gollem.TypeObject && param.AdditionalProperties != nil {
		if path == "" {
			return "(root)", true
		}
		return path, true
	}
	for _, name := range slices.Sorted(maps.Keys(param.Properties)) {
		childPath := name
		if path != "" {
			childPath = path + "." + name
		}
		if found, ok := findAdditionalProperties(param.Properties[name], childPath); ok {
			return found, true
		}
	}
	if param.Items != nil {
		itemsPath := path + "[]"
		if path == "" {
			itemsPath = "(root)[]"
		}
		if found, ok := findAdditionalProperties(param.Items, itemsPath); ok {
			return found, true
		}
	}
	for i, alt := range param.AnyOf {
		base := path
		if base == "" {
			base = "(root)"
		}
		if found, ok := findAdditionalProperties(alt, fmt.Sprintf("%s.anyOf[%d]", base, i)); ok {
			return found, true
		}
	}
	return "", false
}

// CountAnyOf returns the number of parameters in the schema that have AnyOf,
// counting the root, every property, array items, map values and union
// elements at any depth.
//
// Claude structured outputs limits the number of such parameters per request,
// so the Claude client counts them before sending a schema.
func CountAnyOf(param *gollem.Parameter) int {
	if param == nil {
		return 0
	}
	count := 0
	if len(param.AnyOf) > 0 {
		count++
	}
	for _, prop := range param.Properties {
		count += CountAnyOf(prop)
	}
	count += CountAnyOf(param.Items)
	count += CountAnyOf(param.AdditionalProperties)
	for _, alt := range param.AnyOf {
		count += CountAnyOf(alt)
	}
	return count
}

// ConvertAnyOf converts each element of a union with convert, keeping the
// element order, so that every provider emits "anyOf" with the converter it
// uses for any other parameter.
func ConvertAnyOf[T any](alts []*gollem.Parameter, convert func(*gollem.Parameter) T) []T {
	converted := make([]T, len(alts))
	for i, alt := range alts {
		converted[i] = convert(alt)
	}
	return converted
}

// ConvertParameterToJSONSchema converts gollem.Parameter to JSON Schema map
// This is the base conversion without provider-specific modifications
func ConvertParameterToJSONSchema(param *gollem.Parameter) map[string]any {
	// A union has no type of its own; each element carries its type.
	if len(param.AnyOf) > 0 {
		schema := map[string]any{
			"anyOf": ConvertAnyOf(param.AnyOf, ConvertParameterToJSONSchema),
		}
		if param.Description != "" {
			schema["description"] = param.Description
		}
		return schema
	}

	schema := map[string]any{
		"type": string(param.Type),
	}

	if param.Description != "" {
		schema["description"] = param.Description
	}

	if param.Type == gollem.TypeObject && param.Properties != nil {
		props := make(map[string]any)
		for name, prop := range param.Properties {
			props[name] = ConvertParameterToJSONSchema(prop)
		}
		schema["properties"] = props
		schema["additionalProperties"] = false

		// Collect required fields from properties
		if required := CollectRequiredFields(param.Properties); len(required) > 0 {
			schema["required"] = required
		}
	}

	// A map: the value schema of keys not listed in properties. It replaces the
	// "additionalProperties": false set above when an object has both.
	if param.Type == gollem.TypeObject && param.AdditionalProperties != nil {
		schema["additionalProperties"] = ConvertParameterToJSONSchema(param.AdditionalProperties)
	}

	if param.Type == gollem.TypeArray && param.Items != nil {
		schema["items"] = ConvertParameterToJSONSchema(param.Items)
	}

	if param.Enum != nil {
		schema["enum"] = param.Enum
	}

	// Add constraints
	if param.Minimum != nil {
		schema["minimum"] = *param.Minimum
	}
	if param.Maximum != nil {
		schema["maximum"] = *param.Maximum
	}
	if param.MinLength != nil {
		schema["minLength"] = *param.MinLength
	}
	if param.MaxLength != nil {
		schema["maxLength"] = *param.MaxLength
	}
	if param.Pattern != "" {
		schema["pattern"] = param.Pattern
	}
	if param.MinItems != nil {
		schema["minItems"] = *param.MinItems
	}
	if param.MaxItems != nil {
		schema["maxItems"] = *param.MaxItems
	}

	return schema
}

// ConvertParameterToJSONString converts Parameter to a JSON Schema string
// This is used by Claude for embedding schema in system prompt
func ConvertParameterToJSONString(param *gollem.Parameter) (string, error) {
	if param == nil {
		return "", nil
	}

	// Validate schema
	if err := param.Validate(); err != nil {
		return "", goerr.Wrap(err, "invalid response schema")
	}

	// The type and description come from the converted parameter, so a union
	// at the root is written with "anyOf" and no "type".
	schemaObj := ConvertParameterToJSONSchema(param)
	schemaObj["$schema"] = "http://json-schema.org/draft-07/schema#"

	// Marshal to pretty JSON. HTML escaping is disabled because this string is embedded
	// verbatim into the system prompt: with it on, a description containing "<", ">" or "&"
	// reaches the model as a unicode escape sequence instead of the character itself.
	schemaJSON, err := jsonutil.MarshalIndentNoEscape(schemaObj, "", "  ")
	if err != nil {
		return "", goerr.Wrap(err, "failed to marshal schema")
	}

	return string(schemaJSON), nil
}
