package claude

import (
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/gollem-dev/gollem"
	gollemschema "github.com/gollem-dev/gollem/internal/schema"
	"github.com/m-mizutani/goerr/v2"
)

// outputFormat converts a response schema into the output_config.format value
// of a Messages request.
//
// With output_config.format the API decodes the response under the schema, so
// the text is valid JSON matching it. The documented exceptions are a refusal
// (stop_reason "refusal"), a response cut off by max_tokens, and a string enum
// value returned with different capitalization.
//
// Structured outputs accepts additionalProperties only as false, so a schema
// that contains a map (gollem.Parameter.AdditionalProperties) cannot be sent
// this way and is rejected with gollem.ErrUnsupportedSchema before the request
// is sent. See https://platform.claude.com/docs/en/build-with-claude/structured-outputs
// To send such a schema, use a configuration without structured outputs (e.g.
// WithVertexStructuredOutputsDisabled), in which the schema is written into
// the system prompt.
//
// The same page limits "Parameters with union types" to 16: the total number of
// parameters that use anyOf or type arrays across all strict schemas of a
// request. Tools are not sent with strict: true, so the response schema is the
// only strict schema, and a schema with more than 16 parameters with AnyOf is
// rejected with gollem.ErrUnsupportedSchema instead of a 400 from the API.
func outputFormat(param *gollem.Parameter) (anthropic.JSONOutputFormatParam, error) {
	if err := param.Validate(); err != nil {
		return anthropic.JSONOutputFormatParam{}, goerr.Wrap(err, "invalid response schema")
	}
	if path, found := gollemschema.FindAdditionalProperties(param); found {
		return anthropic.JSONOutputFormatParam{}, goerr.Wrap(gollem.ErrUnsupportedSchema,
			fmt.Sprintf("map at %q cannot be sent as Claude structured outputs, which require additionalProperties to be false", path),
			goerr.V("path", path))
	}
	if count := gollemschema.CountAnyOf(param); count > maxUnionParameters {
		return anthropic.JSONOutputFormatParam{}, goerr.Wrap(gollem.ErrUnsupportedSchema,
			fmt.Sprintf("schema has %d parameters with anyOf, more than the %d that Claude structured outputs accepts", count, maxUnionParameters),
			goerr.V("count", count), goerr.V("limit", maxUnionParameters))
	}
	return anthropic.JSONOutputFormatParam{Schema: outputSchema(param)}, nil
}

// maxUnionParameters is the "Parameters with union types" limit of Claude
// structured outputs.
const maxUnionParameters = 16

// outputSchema converts a parameter into the JSON Schema subset that
// structured outputs accepts.
//
// Structured outputs rejects numeric bounds, string length and pattern
// constraints, maxItems and a minItems above 1, and requires
// additionalProperties to be false on every object. The rejected constraints
// are removed from the schema and restated in the description, so the model is
// still told about them; the API does not enforce them and this package does
// not validate the response against them.
//
// A union is sent as "anyOf" with each element converted by this function, so
// each object element gets additionalProperties false and its own rejected
// constraints restated in its description.
func outputSchema(param *gollem.Parameter) map[string]any {
	if len(param.AnyOf) > 0 {
		schema := map[string]any{
			"anyOf": gollemschema.ConvertAnyOf(param.AnyOf, outputSchema),
		}
		if param.Default != nil {
			schema["default"] = param.Default
		}
		if param.Description != "" {
			schema["description"] = param.Description
		}
		return schema
	}

	schema := map[string]any{
		"type": string(param.Type),
	}

	if param.Type == gollem.TypeObject && param.Properties != nil {
		props := make(map[string]any, len(param.Properties))
		for name, prop := range param.Properties {
			props[name] = outputSchema(prop)
		}
		schema["properties"] = props
		schema["additionalProperties"] = false
		if required := gollemschema.CollectRequiredFields(param.Properties); len(required) > 0 {
			schema["required"] = required
		}
	}

	if param.Type == gollem.TypeArray && param.Items != nil {
		schema["items"] = outputSchema(param.Items)
	}

	if param.Enum != nil {
		schema["enum"] = param.Enum
	}

	if param.Default != nil {
		schema["default"] = param.Default
	}

	// The order is fixed so that identical schemas produce identical request
	// bytes; a changed schema invalidates the compiled grammar and prompt cache.
	var constraints []string
	if param.Minimum != nil {
		constraints = append(constraints, fmt.Sprintf("minimum: %v", *param.Minimum))
	}
	if param.Maximum != nil {
		constraints = append(constraints, fmt.Sprintf("maximum: %v", *param.Maximum))
	}
	if param.MinLength != nil {
		constraints = append(constraints, fmt.Sprintf("minLength: %d", *param.MinLength))
	}
	if param.MaxLength != nil {
		constraints = append(constraints, fmt.Sprintf("maxLength: %d", *param.MaxLength))
	}
	if param.Pattern != "" {
		constraints = append(constraints, fmt.Sprintf("pattern: %s", param.Pattern))
	}
	if param.MinItems != nil {
		if *param.MinItems <= 1 {
			schema["minItems"] = *param.MinItems
		} else {
			constraints = append(constraints, fmt.Sprintf("minItems: %d", *param.MinItems))
		}
	}
	if param.MaxItems != nil {
		constraints = append(constraints, fmt.Sprintf("maxItems: %d", *param.MaxItems))
	}

	description := param.Description
	if len(constraints) > 0 {
		note := "{" + strings.Join(constraints, ", ") + "}"
		if description != "" {
			description += "\n\n" + note
		} else {
			description = note
		}
	}
	if description != "" {
		schema["description"] = description
	}

	return schema
}
