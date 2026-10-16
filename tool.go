package gollem

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/m-mizutani/goerr/v2"
)

// ToolSpec is the specification of a tool.
type ToolSpec struct {
	Name        string
	Description string
	Parameters  map[string]*Parameter
}

// ValidateArgs validates the given arguments against the tool's parameter specifications.
// It checks all parameters and collects all validation errors.
// Returns nil if all arguments are valid.
func (s *ToolSpec) ValidateArgs(args map[string]any) error {
	// Sort parameter names for deterministic error ordering
	names := make([]string, 0, len(s.Parameters))
	for name := range s.Parameters {
		names = append(names, name)
	}
	slices.Sort(names)

	var errs []error
	for _, name := range names {
		param := s.Parameters[name]
		if err := param.ValidateValue(name, args[name]); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return &toolArgsValidationError{
			toolName: s.Name,
			errs:     errs,
		}
	}
	return nil
}

// toolArgsValidationError holds multiple argument validation errors for a tool.
// It formats a structured error message that helps LLMs understand and correct invalid arguments.
type toolArgsValidationError struct {
	toolName string
	errs     []error
}

func (e *toolArgsValidationError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "Tool argument validation failed for %q:\n", e.toolName)
	for _, err := range e.errs {
		fmt.Fprintf(&b, "  - %s\n", err.Error())
	}
	b.WriteString("Please correct the arguments and retry the tool call.")
	return b.String()
}

func (e *toolArgsValidationError) Unwrap() error {
	return ErrToolArgsValidation
}

// Validate validates the tool specification.
func (s *ToolSpec) Validate() error {
	eb := goerr.NewBuilder(goerr.V("tool", s))
	if s.Name == "" {
		return eb.Wrap(ErrInvalidTool, "name is required")
	}

	paramNames := make(map[string]struct{})
	for _, name := range slices.Sorted(maps.Keys(s.Parameters)) {
		if _, ok := paramNames[name]; ok {
			return eb.Wrap(ErrInvalidTool, "duplicate parameter name", goerr.V("name", name))
		}
		paramNames[name] = struct{}{}

		if err := s.Parameters[name].Validate(); err != nil {
			// goerr keeps a single cause, so ErrInvalidTool is joined with the
			// cause; wrapping only err would break errors.Is(err, ErrInvalidTool).
			return eb.Wrap(fmt.Errorf("%w: %w", ErrInvalidTool, err), fmt.Sprintf("invalid parameter %q", name))
		}
	}

	return nil
}

// ParameterType is the type of a parameter.
type ParameterType string

const (
	TypeString  ParameterType = "string"
	TypeNumber  ParameterType = "number"
	TypeInteger ParameterType = "integer"
	TypeBoolean ParameterType = "boolean"
	TypeArray   ParameterType = "array"
	TypeObject  ParameterType = "object"
)

// Parameter is a parameter of a tool.
type Parameter struct {
	// Title is the user friendly  of the parameter. It's optional.
	Title string

	// Type is the type of the parameter. It is required when AnyOf is nil and
	// must be empty when AnyOf is set.
	Type ParameterType

	// Description is the description of the parameter. It's optional.
	Description string

	// Required indicates if this parameter is required.
	Required bool

	// Enum is the enum of the parameter. It's optional.
	Enum []string

	// Properties is the properties of the parameter. It's used for object type.
	Properties map[string]*Parameter

	// Items is the items of the parameter. It's used for array type.
	Items *Parameter

	// AdditionalProperties is the schema of the values of keys that are not
	// listed in Properties. It is used for object type and represents a Go map,
	// e.g. map[string]int is {Type: object, AdditionalProperties: {Type: integer}}.
	//
	// Not every way of sending a schema to an LLM accepts it:
	//   - Tool definitions: accepted by Claude, OpenAI and Gemini.
	//   - Response schemas: accepted by OpenAI without strict mode, by Gemini,
	//     and by Claude when the schema is written into the system prompt
	//     (models or configurations without structured outputs).
	//   - Rejected with ErrUnsupportedSchema before calling the API by Claude
	//     structured outputs and OpenAI strict mode, because both require
	//     additionalProperties to be false on every object.
	AdditionalProperties *Parameter

	// AnyOf lists the alternative schemas a value may match; the value is valid
	// when it matches at least one. When AnyOf is set, the parameter itself
	// carries no type of its own: Type, Properties, Items, AdditionalProperties,
	// Enum and the constraints must be empty, and only Title, Description,
	// Required and Default may be set. AnyOf needs at least two elements.
	//
	// It is sent as JSON Schema "anyOf" at any depth. A typical use is an array
	// whose items are objects of several kinds, each kind an element of AnyOf
	// with a single-value Enum property (e.g. "kind") that tells them apart.
	// Properties are sent in ascending name order and constrained decoding
	// writes them in that order, so give that property a name that comes first
	// among the required properties of every element; otherwise the model can
	// only produce the elements in which it comes first.
	//
	// Where it is accepted:
	//   - Tool definitions: accepted by Claude, OpenAI, Gemini and Ollama.
	//   - Response schemas: accepted by Claude (structured outputs and the
	//     system-prompt schema), OpenAI (with and without strict mode), Gemini
	//     and Ollama, whose server turns "anyOf" into a grammar alternation.
	//   - Rejected with ErrUnsupportedSchema before calling the API:
	//       - Claude structured outputs, when the schema has more than 16
	//         parameters with AnyOf; the API counts them across all strict
	//         schemas and rejects a request above that limit.
	//       - OpenAI strict mode, when AnyOf is set on the root of the
	//         response schema; the root must be an object.
	AnyOf []*Parameter

	// Number constraints
	Minimum *float64
	Maximum *float64

	// String constraints
	MinLength *int
	MaxLength *int
	Pattern   string

	// Array constraints
	MinItems *int
	MaxItems *int

	// Default value
	Default any
}

// Validate validates the parameter.
func (p *Parameter) Validate() error {
	eb := goerr.NewBuilder(goerr.V("parameter", p))

	if p.AnyOf != nil {
		return p.validateAnyOf()
	}

	// Type is required
	if p.Type == "" {
		return eb.Wrap(ErrInvalidParameter, "type is required")
	}

	// Validate parameter type
	switch p.Type {
	case TypeString, TypeNumber, TypeInteger, TypeBoolean, TypeArray, TypeObject:
		// Valid type
	default:
		return eb.Wrap(ErrInvalidParameter, "invalid parameter type", goerr.V("type", p.Type))
	}

	// An object needs named properties, a value schema for its other keys
	// (a map), or both.
	if p.Type == TypeObject {
		if p.Properties == nil && p.AdditionalProperties == nil {
			return eb.Wrap(ErrInvalidParameter, "properties or additionalProperties is required for object type")
		}

		// Check for duplicate property names
		propNames := make(map[string]struct{})
		for name := range p.Properties {
			if _, ok := propNames[name]; ok {
				return eb.Wrap(ErrInvalidParameter, "duplicate property name", goerr.V("name", name))
			}
			propNames[name] = struct{}{}
		}

		// Validate nested properties in name order so that the reported
		// property does not depend on map iteration order.
		for _, name := range slices.Sorted(maps.Keys(p.Properties)) {
			if err := p.Properties[name].Validate(); err != nil {
				return eb.Wrap(err, fmt.Sprintf("invalid property %q", name))
			}
		}

		if p.AdditionalProperties != nil {
			if err := p.AdditionalProperties.Validate(); err != nil {
				return eb.Wrap(err, "invalid additionalProperties")
			}
		}
	}

	// Items is required for array type
	if p.Type == TypeArray {
		if p.Items == nil {
			return eb.Wrap(ErrInvalidParameter, "items is required for array type")
		}
		// Validate items
		if err := p.Items.Validate(); err != nil {
			return eb.Wrap(err, "invalid items")
		}
	}

	// Validate number constraints
	if p.Type == TypeNumber || p.Type == TypeInteger {
		if p.Minimum != nil && p.Maximum != nil && *p.Minimum > *p.Maximum {
			return eb.Wrap(ErrInvalidParameter, "minimum must be less than or equal to maximum")
		}
	}

	// Validate string constraints
	if p.Type == TypeString {
		if p.MinLength != nil && p.MaxLength != nil && *p.MinLength > *p.MaxLength {
			return eb.Wrap(ErrInvalidParameter, "minLength must be less than or equal to maxLength")
		}
		if p.Pattern != "" {
			if _, err := regexp.Compile(p.Pattern); err != nil {
				return eb.Wrap(ErrInvalidParameter, "invalid pattern", goerr.V("pattern", p.Pattern))
			}
		}
	}

	// Validate array constraints
	if p.Type == TypeArray {
		if p.MinItems != nil && p.MaxItems != nil && *p.MinItems > *p.MaxItems {
			return eb.Wrap(ErrInvalidParameter, "minItems must be less than or equal to maxItems")
		}
	}

	return nil
}

// validateAnyOf validates a parameter that has AnyOf. The union has no type of
// its own, so every field that describes a type must be empty; the elements
// carry the types.
func (p *Parameter) validateAnyOf() error {
	eb := goerr.NewBuilder(goerr.V("parameter", p))

	var typed []string
	if p.Type != "" {
		typed = append(typed, "type")
	}
	if p.Properties != nil {
		typed = append(typed, "properties")
	}
	if p.Items != nil {
		typed = append(typed, "items")
	}
	if p.AdditionalProperties != nil {
		typed = append(typed, "additionalProperties")
	}
	if p.Enum != nil {
		typed = append(typed, "enum")
	}
	if p.Minimum != nil {
		typed = append(typed, "minimum")
	}
	if p.Maximum != nil {
		typed = append(typed, "maximum")
	}
	if p.MinLength != nil {
		typed = append(typed, "minLength")
	}
	if p.MaxLength != nil {
		typed = append(typed, "maxLength")
	}
	if p.Pattern != "" {
		typed = append(typed, "pattern")
	}
	if p.MinItems != nil {
		typed = append(typed, "minItems")
	}
	if p.MaxItems != nil {
		typed = append(typed, "maxItems")
	}
	if len(typed) > 0 {
		return eb.Wrap(ErrInvalidParameter,
			fmt.Sprintf("anyOf must not be combined with %s", strings.Join(typed, ", ")),
			goerr.V("fields", typed))
	}

	if len(p.AnyOf) < 2 {
		return eb.Wrap(ErrInvalidParameter, "anyOf needs at least two elements",
			goerr.V("count", len(p.AnyOf)))
	}

	for i, alt := range p.AnyOf {
		if alt == nil {
			return eb.Wrap(ErrInvalidParameter, fmt.Sprintf("anyOf[%d] must not be nil", i))
		}
		if err := alt.Validate(); err != nil {
			return eb.Wrap(err, fmt.Sprintf("invalid anyOf[%d]", i), goerr.V("index", i))
		}
	}
	return nil
}

// ValidateValue validates a value against this parameter's specification.
// It checks required, type, enum, and constraint validations.
// Returns nil if the value is valid, or an error describing the validation failure.
func (p *Parameter) ValidateValue(name string, value any) error {
	eb := goerr.NewBuilder(goerr.V("parameter", name))

	// Check required
	if value == nil {
		if p.Required {
			return eb.Wrap(ErrInvalidParameter, "required parameter missing")
		}
		return nil // Optional parameter with no value is valid
	}

	if len(p.AnyOf) > 0 {
		return p.validateAnyOfValue(name, value)
	}

	// Type validation
	switch p.Type {
	case TypeString:
		s, ok := value.(string)
		if !ok {
			return eb.Wrap(ErrInvalidParameter, "expected string type", goerr.V("actual", value))
		}
		// Enum validation
		if len(p.Enum) > 0 && !slices.Contains(p.Enum, s) {
			return eb.Wrap(ErrInvalidParameter, "value not in enum", goerr.V("value", s), goerr.V("enum", p.Enum))
		}
		// String length validation
		if p.MinLength != nil && len(s) < *p.MinLength {
			return eb.Wrap(ErrInvalidParameter, "string too short", goerr.V("minLength", *p.MinLength), goerr.V("actual", len(s)))
		}
		if p.MaxLength != nil && len(s) > *p.MaxLength {
			return eb.Wrap(ErrInvalidParameter, "string too long", goerr.V("maxLength", *p.MaxLength), goerr.V("actual", len(s)))
		}
		// Pattern validation
		if p.Pattern != "" {
			matched, err := regexp.MatchString(p.Pattern, s)
			if err != nil {
				return eb.Wrap(err, "pattern matching failed")
			}
			if !matched {
				return eb.Wrap(ErrInvalidParameter, "string does not match pattern", goerr.V("pattern", p.Pattern))
			}
		}

	case TypeNumber:
		var n float64
		switch v := value.(type) {
		case float64:
			n = v
		case float32:
			n = float64(v)
		case int:
			n = float64(v)
		case int64:
			n = float64(v)
		case json.Number:
			// An argument too wide for float64 is decoded as json.Number to keep its exact
			// value; the range check below is still done in float64, which is enough to
			// compare against Minimum and Maximum.
			f, err := v.Float64()
			if err != nil {
				return eb.Wrap(ErrInvalidParameter, "expected number type", goerr.V("actual", value))
			}
			n = f
		default:
			return eb.Wrap(ErrInvalidParameter, "expected number type", goerr.V("actual", value))
		}
		if p.Minimum != nil && n < *p.Minimum {
			return eb.Wrap(ErrInvalidParameter, "number too small", goerr.V("minimum", *p.Minimum), goerr.V("actual", n))
		}
		if p.Maximum != nil && n > *p.Maximum {
			return eb.Wrap(ErrInvalidParameter, "number too large", goerr.V("maximum", *p.Maximum), goerr.V("actual", n))
		}

	case TypeInteger:
		var n int64
		switch v := value.(type) {
		case int:
			n = int64(v)
		case int64:
			n = v
		case float64:
			if v != float64(int64(v)) {
				return eb.Wrap(ErrInvalidParameter, "expected integer type, got float", goerr.V("actual", v))
			}
			n = int64(v)
		case json.Number:
			// An integer too wide for float64 is decoded as json.Number to keep its exact
			// value, so it is parsed as an int64 rather than through float64. A literal
			// that does not fit an int64 is out of range for this parameter; the parse
			// error is kept so the reason is visible in the error chain.
			i, err := v.Int64()
			if err != nil {
				return eb.Wrap(ErrInvalidParameter, "expected integer type",
					goerr.V("actual", value), goerr.V("parse_error", err))
			}
			n = i
		default:
			return eb.Wrap(ErrInvalidParameter, "expected integer type", goerr.V("actual", value))
		}
		if p.Minimum != nil && float64(n) < *p.Minimum {
			return eb.Wrap(ErrInvalidParameter, "integer too small", goerr.V("minimum", *p.Minimum), goerr.V("actual", n))
		}
		if p.Maximum != nil && float64(n) > *p.Maximum {
			return eb.Wrap(ErrInvalidParameter, "integer too large", goerr.V("maximum", *p.Maximum), goerr.V("actual", n))
		}

	case TypeBoolean:
		if _, ok := value.(bool); !ok {
			return eb.Wrap(ErrInvalidParameter, "expected boolean type", goerr.V("actual", value))
		}

	case TypeArray:
		arr, ok := value.([]any)
		if !ok {
			return eb.Wrap(ErrInvalidParameter, "expected array type", goerr.V("actual", value))
		}
		if p.MinItems != nil && len(arr) < *p.MinItems {
			return eb.Wrap(ErrInvalidParameter, "array too short", goerr.V("minItems", *p.MinItems), goerr.V("actual", len(arr)))
		}
		if p.MaxItems != nil && len(arr) > *p.MaxItems {
			return eb.Wrap(ErrInvalidParameter, "array too long", goerr.V("maxItems", *p.MaxItems), goerr.V("actual", len(arr)))
		}
		// Validate each item if Items schema is defined
		if p.Items != nil {
			for i, item := range arr {
				if err := p.Items.ValidateValue(name+"["+strconv.Itoa(i)+"]", item); err != nil {
					return err
				}
			}
		}

	case TypeObject:
		obj, ok := value.(map[string]any)
		if !ok {
			return eb.Wrap(ErrInvalidParameter, "expected object type", goerr.V("actual", value))
		}
		// Validate each property if Properties schema is defined
		for _, propName := range slices.Sorted(maps.Keys(p.Properties)) {
			if err := p.Properties[propName].ValidateValue(name+"."+propName, obj[propName]); err != nil {
				return err
			}
		}
		// Validate the values of the other keys against the map value schema
		if p.AdditionalProperties != nil {
			for _, key := range slices.Sorted(maps.Keys(obj)) {
				if _, listed := p.Properties[key]; listed {
					continue
				}
				// A present key holds a value, so null does not mean "omitted"
				// here; the value schema has no null type, and decoding null into
				// a non-pointer Go value would silently yield the zero value.
				if obj[key] == nil {
					return goerr.Wrap(ErrInvalidParameter, "map value must not be null",
						goerr.V("parameter", name+"."+key))
				}
				if err := p.AdditionalProperties.ValidateValue(name+"."+key, obj[key]); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

// validateAnyOfValue accepts value when it matches at least one element of
// AnyOf. When none matches, the error message lists every element's failure,
// because the message is what tool argument validation returns to the model
// and the model needs each reason to correct the value.
func (p *Parameter) validateAnyOfValue(name string, value any) error {
	failures := make([]string, 0, len(p.AnyOf))
	for i, alt := range p.AnyOf {
		err := alt.ValidateValue(name, value)
		if err == nil {
			return nil
		}
		failure := fmt.Sprintf("anyOf[%d]: %s", i, err.Error())
		// Each element reports the path of the value that failed; nested
		// failures name a property or item below name, which the message alone
		// would not show.
		if path, ok := goerr.Values(err)["parameter"].(string); ok && path != name {
			failure = fmt.Sprintf("anyOf[%d]: %s at %q", i, err.Error(), path)
		}
		failures = append(failures, failure)
	}
	return goerr.Wrap(ErrInvalidParameter,
		fmt.Sprintf("value matches none of anyOf (%s)", strings.Join(failures, "; ")),
		goerr.V("parameter", name), goerr.V("failures", failures))
}

// Tool is specification and execution of an action that can be called by the LLM.
type Tool interface {
	// Spec returns the specification of the tool. It's called when starting a LLM chat session in Prompt().
	Spec() ToolSpec

	// Run is the execution of the tool.
	// It's called when receiving a tool call from the LLM. Even if the method returns an error, the tool execution is not aborted. Error will be passed to LLM as a response. If you want to abort the tool execution, you need to return an error from the callback function of WithToolErrorHook().
	// Special case: If the tool returns ErrExitConversation, the conversation loop will be terminated normally and the session will be completed successfully.
	Run(ctx context.Context, args map[string]any) (map[string]any, error)
}

// ToolSet is a set of tools.
// It's useful for providing a set of tools to the LLM.
type ToolSet interface {
	// Specs returns the specifications of the tools.
	Specs(ctx context.Context) ([]ToolSpec, error)

	// Run is the execution of the tool.
	// It's called when receiving a tool call from the LLM.
	Run(ctx context.Context, name string, args map[string]any) (map[string]any, error)
}
