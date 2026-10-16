# Structured Output with JSON Schema

gollem provides comprehensive support for structured output using JSON Schema, allowing you to constrain LLM responses to specific formats. This ensures predictable, parseable outputs for data extraction, API responses, and structured data generation.

## Overview

By defining a JSON Schema, you can:
- **Guarantee response format**: LLM outputs always conform to your schema
- **Extract structured data**: Parse unstructured text into structured objects
- **Type safety**: Define expected types, constraints, and validation rules
- **Multi-provider support**: Works consistently across OpenAI, Claude, and Gemini

## Basic Usage

### 1. Define a Response Schema

```go
schema := &gollem.Parameter{
	Title:       "UserProfile",
	Description: "Structured user profile information",
	Type:        gollem.TypeObject,
	Properties: map[string]*gollem.Parameter{
		"name": {
			Type:        gollem.TypeString,
			Description: "Full name of the user",
		},
		"age": {
			Type:        gollem.TypeInteger,
			Description: "Age in years",
			Minimum:     Ptr(0.0),
			Maximum:     Ptr(150.0),
		},
		"email": {
			Type:        gollem.TypeString,
			Description: "Email address",
		},
	},
	Required: []string{"name", "email"},
}
```

### 2. Create Session with Schema

```go
session, err := client.NewSession(ctx,
	gollem.WithSessionContentType(gollem.ContentTypeJSON),
	gollem.WithSessionResponseSchema(schema),
)
if err != nil {
	return err
}
```

### 3. Generate Structured Content

```go
resp, err := session.Generate(ctx, []gollem.Input{
	gollem.Text("Extract user info: John Doe, 30 years old, john@example.com")})
if err != nil {
	return err
}

// Response is guaranteed to be valid JSON matching the schema
var profile map[string]any
json.Unmarshal([]byte(resp.Texts[0]), &profile)
fmt.Printf("Name: %s, Age: %v, Email: %s\n",
	profile["name"], profile["age"], profile["email"])
```

## One-Shot Typed Query with `Query[T]()`

For the common pattern of "send prompt, get structured data back", use the generic `Query[T]()` function. It combines schema generation, session creation, LLM call, and JSON parsing into a single call with automatic retry on parse failures:

```go
type UserProfile struct {
	Name  string `json:"name" description:"Full name of the user" required:"true"`
	Age   int    `json:"age" description:"Age in years" min:"0" max:"150"`
	Email string `json:"email" description:"Email address" required:"true"`
}

result, err := gollem.Query[UserProfile](ctx, client,
	"Extract user info: John Doe, 30 years old, john@example.com",
	gollem.WithQuerySystemPrompt("You are a data extraction assistant."),
)
if err != nil {
	return err
}

fmt.Printf("Name: %s, Age: %d, Email: %s\n",
	result.Data.Name, result.Data.Age, result.Data.Email)
fmt.Printf("Tokens used: %d input, %d output\n",
	result.InputToken, result.OutputToken)
```

### Query Options

| Option | Description |
|--------|-------------|
| `WithQuerySystemPrompt(string)` | Set the system prompt for the query |
| `WithQueryHistory(*History)` | Provide conversation history |
| `WithQueryMaxRetry(int)` | Maximum retries on JSON parse failure or on a response without text (default: 3) |

### Retries

`Query` and `SessionQuery` send another request in these cases, up to the
maximum number of retries:

- The response text is not valid JSON, or does not match the schema. The next
  message gives the error and the response text, and asks for valid JSON.
- The response has no text and no function call, and the model ended the
  generation on its own: `FinishReason` is `end_turn` (Claude), `STOP`
  (Gemini), `stop` (OpenAI Chat Completions and Ollama) or `completed` (OpenAI
  Responses API). This happens, for example, when a Claude model answers with
  only a thinking block whose text was omitted. The same request would be
  answered the same way, so the next message asks the model to continue. The
  response stays in the session history as received; a response without any
  content is not added, so the new message directly follows the previous user
  message.

A response without text that does not meet these conditions, for example one
that stopped at the token limit, is returned as an error without a retry. The
error carries the goerr values `attempt`, `finish_reason`, `other_block_types`,
`output_tokens` and `function_calls` (the number of function calls). A
response with a refusal is also returned as an error without a retry; see
[Refusal Details](llm.md#refusal-details).

## Session-Based Typed Query with `SessionQuery[T]()`

`SessionQuery[T]()` is like `Query[T]()` but operates on an **existing session** instead of creating a new one. This preserves the conversation history, so the LLM can use context from prior exchanges to answer the structured query.

```go
// Create a session and have a conversation
session, _ := client.NewSession(ctx)
session.Generate(ctx, []gollem.Input{gollem.Text("My name is Alice and I'm 30.")})

// Now ask a structured question — history is preserved
type UserInfo struct {
    Name string `json:"name" required:"true"`
    Age  int    `json:"age"`
}

result, err := gollem.SessionQuery[UserInfo](ctx, session, "Who am I?")
// result.Data.Name == "Alice", result.Data.Age == 30
```

Internally, `SessionQuery` passes a per-call `GenerateOption` with the response schema so the session's default configuration is unchanged.

### SessionQuery Options

| Option | Description |
|--------|-------------|
| `WithSessionQueryMaxRetry(int)` | Maximum retries on JSON parse failure or on a response without text (default: 3) |

## Per-Call Generate Options

`Generate` and `Stream` accept optional `GenerateOption` values that override session-level defaults for a single call only. The session's configuration is not modified.

```go
// Session created with default text mode
session, _ := client.NewSession(ctx)

// Limit the output length for one call
resp, _ := session.Generate(ctx, inputs, gollem.WithMaxTokens(256))

// Force JSON output with a schema for one call
schema, _ := gollem.ToSchema(MyStruct{})
resp, _ = session.Generate(ctx, inputs, gollem.WithGenerateResponseSchema(schema))
```

### Available Options

| Option | Description |
|--------|-------------|
| `WithMaxTokens(int)` | Override max tokens for this call |
| `WithGenerateResponseSchema(*Parameter)` | Force JSON output with the given schema for this call |
| `WithToolCallsDisabled()` | Forbid tool calls for this call while still sending the session's tools |

There is no per-call option for temperature or top-p. Recent Claude, OpenAI, and Gemini models reject these parameters or no longer support them. To set them for a model that accepts them, use the client option of each provider, such as `ollama.WithTemperature` (see [LLM Providers](llm.md)).

When `WithGenerateResponseSchema` is set, the provider automatically switches to JSON output mode for that call (e.g., OpenAI sets `ResponseFormat` to JSON Schema, Gemini sets `ResponseMIMEType` to `application/json`, Claude sets `output_config.format` on models that support structured outputs; see [Provider-Specific Behavior](#provider-specific-behavior)).

`WithToolCallsDisabled` sends `tool_choice: "none"` to Claude and OpenAI and the function calling mode `NONE` to Gemini. The model cannot call a tool in that call, but the tool definitions are still sent, so the request carries the same tool list as the calls before it. Claude rejects a history containing thinking blocks when the tool list differs from the one they were produced with; removing the tools instead of using this option causes that rejection. When the session has no tools, the option sends nothing.

```go
// After a tool loop, ask for a structured answer over the same history
// without letting the model call another tool.
resp, _ = session.Generate(ctx,
	[]gollem.Input{gollem.Text("Report the result as JSON.")},
	gollem.WithToolCallsDisabled(),
	gollem.WithGenerateResponseSchema(schema),
)
```

## Schema Parameter Types

gollem supports the following JSON Schema types:

### Basic Types

```go
// String
&gollem.Parameter{
	Type:        gollem.TypeString,
	Description: "A text value",
	Pattern:     "^[a-z]+$",        // Optional regex pattern
	MinLength:   Ptr(1),             // Optional minimum length
	MaxLength:   Ptr(100),           // Optional maximum length
}

// Integer
&gollem.Parameter{
	Type:        gollem.TypeInteger,
	Description: "An integer value",
	Minimum:     Ptr(0.0),           // Optional minimum value
	Maximum:     Ptr(100.0),         // Optional maximum value
}

// Number (floating point)
&gollem.Parameter{
	Type:        gollem.TypeNumber,
	Description: "A numeric value",
	Minimum:     Ptr(0.0),
	Maximum:     Ptr(1.0),
}

// Boolean
&gollem.Parameter{
	Type:        gollem.TypeBoolean,
	Description: "A true/false value",
}
```

### Complex Types

```go
// Object
&gollem.Parameter{
	Type: gollem.TypeObject,
	Properties: map[string]*gollem.Parameter{
		"field1": {Type: gollem.TypeString},
		"field2": {Type: gollem.TypeInteger},
	},
	Required: []string{"field1"},  // Specify required fields
}

// Array
&gollem.Parameter{
	Type: gollem.TypeArray,
	Items: &gollem.Parameter{
		Type: gollem.TypeString,
	},
	MinItems: Ptr(1),   // Optional minimum array length
	MaxItems: Ptr(10),  // Optional maximum array length
}

// Enum (restricted values)
&gollem.Parameter{
	Type: gollem.TypeString,
	Enum: []any{"active", "inactive", "pending"},
	Description: "User status",
}

// Map (any keys, values of one schema), e.g. map[string]int
&gollem.Parameter{
	Type: gollem.TypeObject,
	AdditionalProperties: &gollem.Parameter{
		Type: gollem.TypeInteger,
	},
}

// Union (the value matches at least one of the schemas), sent as "anyOf"
&gollem.Parameter{
	AnyOf: []*gollem.Parameter{
		{Type: gollem.TypeString},
		{Type: gollem.TypeInteger},
	},
}
```

Not every way of sending a schema accepts a map; see [Maps](#maps). A union has limits of its own; see [Unions](#unions).

### Unions

`AnyOf` declares a value that may take one of several shapes. A typical use is an array whose items are objects of different kinds: each kind is its own object schema, and a property with a single-value `Enum` (here `kind`) tells them apart. Under Claude structured outputs or OpenAI strict mode, the model then cannot set a field on a kind that does not define it, which a single object carrying every kind's fields cannot prevent.

```go
blocks := &gollem.Parameter{
	Type: gollem.TypeArray,
	Items: &gollem.Parameter{
		Description: "A block of the document body",
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
					"tone": {Type: gollem.TypeString, Enum: []string{"info", "warning"}, Required: true},
				},
			},
		},
	},
}
```

- A parameter with `AnyOf` has no type of its own. `Type`, `Properties`, `Items`, `AdditionalProperties`, `Enum` and the constraints must be empty; only `Title`, `Description`, `Required` and `Default` may be set. `AnyOf` needs at least two elements. `Parameter.Validate` returns `gollem.ErrInvalidParameter` otherwise, and the message names the index of an invalid element, for example `invalid anyOf[1]`.
- A value is valid when it matches at least one element. When it matches none, `Parameter.ValidateValue` returns `gollem.ErrInvalidParameter` whose message lists each element's failure, for example `anyOf[0]: value not in enum: invalid parameter at "block.kind"`. Tool argument validation returns this message to the model.
- gollem sends the properties of an object in ascending name order, and constrained decoding (Claude structured outputs, OpenAI strict mode) writes them in that order; Claude writes the required properties first. Name the property that tells the kinds apart so that it comes first among the required properties of every kind, as `kind` does above. If another required property sorts before it, for example `color`, the model writes `"kind"` first only for the kinds where `kind` is the first property, and cannot choose the other kinds. With OpenAI strict mode, a `color` property in the callout kind above made the model return only paragraphs.
- `ToSchema` does not produce `AnyOf`, because a Go struct cannot declare a union. Build the schema by hand, and decode the reply yourself, for example into a struct that holds every kind's fields and switches on `kind`.

| Where the schema is sent | Union |
|---|---|
| Tool definitions (Claude, OpenAI, Gemini, Ollama) | Sent |
| Response schema, Ollama | Sent (as `format`) |
| Response schema, OpenAI without strict mode (the default) | Sent |
| Response schema, Gemini | Sent (in `genai.Schema.AnyOf`) |
| Response schema, Claude without structured outputs (schema in the system prompt) | Sent |
| Response schema, Claude structured outputs | Sent; more than 16 parameters with `AnyOf` are rejected with `gollem.ErrUnsupportedSchema` |
| Response schema, OpenAI strict mode | Sent below the root; `AnyOf` on the root is rejected with `gollem.ErrUnsupportedSchema` |

Claude structured outputs accepts at most 16 parameters that use `anyOf` in a request ("Parameters with union types" in the [structured outputs limits](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)), so gollem counts them and rejects a larger schema before calling the API. OpenAI strict mode requires the root of the schema to be an object that does not use `anyOf`. In both, each object element is sent with `additionalProperties: false`, and in OpenAI strict mode every property of an object element is listed in `required`, as for any other object.

## Advanced Examples

### Nested Objects

```go
schema := &gollem.ResponseSchema{
	Name: "Organization",
	Schema: &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"name": {
				Type: gollem.TypeString,
			},
			"address": {
				Type: gollem.TypeObject,
				Properties: map[string]*gollem.Parameter{
					"street": {Type: gollem.TypeString},
					"city":   {Type: gollem.TypeString},
					"country": {Type: gollem.TypeString},
					"zipcode": {
						Type:    gollem.TypeString,
						Pattern: "^[0-9]{5}$",
					},
				},
				Required: []string{"city", "country"},
			},
		},
		Required: []string{"name", "address"},
	},
}
```

### Arrays of Objects

```go
schema := &gollem.ResponseSchema{
	Name: "EmployeeList",
	Schema: &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"employees": {
				Type: gollem.TypeArray,
				Items: &gollem.Parameter{
					Type: gollem.TypeObject,
					Properties: map[string]*gollem.Parameter{
						"name": {
							Type: gollem.TypeString,
						},
						"position": {
							Type: gollem.TypeString,
							Enum: []any{"engineer", "manager", "designer"},
						},
						"salary": {
							Type:    gollem.TypeInteger,
							Minimum: Ptr(0.0),
						},
					},
					Required: []string{"name", "position"},
				},
				MinItems: Ptr(1),
			},
		},
	},
}
```

### Complex Data Extraction

```go
schema := &gollem.ResponseSchema{
	Name:        "SecurityAlert",
	Description: "Structured security alert analysis",
	Schema: &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"severity": {
				Type: gollem.TypeString,
				Enum: []any{"low", "medium", "high", "critical"},
			},
			"threat_type": {
				Type: gollem.TypeString,
			},
			"affected_systems": {
				Type: gollem.TypeArray,
				Items: &gollem.Parameter{
					Type: gollem.TypeString,
				},
			},
			"indicators": {
				Type: gollem.TypeObject,
				Properties: map[string]*gollem.Parameter{
					"ip_addresses": {
						Type: gollem.TypeArray,
						Items: &gollem.Parameter{Type: gollem.TypeString},
					},
					"domains": {
						Type: gollem.TypeArray,
						Items: &gollem.Parameter{Type: gollem.TypeString},
					},
					"file_hashes": {
						Type: gollem.TypeArray,
						Items: &gollem.Parameter{Type: gollem.TypeString},
					},
				},
			},
			"recommended_actions": {
				Type: gollem.TypeArray,
				Items: &gollem.Parameter{
					Type: gollem.TypeString,
				},
				MinItems: Ptr(1),
			},
		},
		Required: []string{"severity", "threat_type", "recommended_actions"},
	},
}
```

## Creating Schemas from Go Structs

### ToSchema

Instead of manually constructing `gollem.Parameter` objects, you can automatically generate schemas from Go struct types using struct field tags:

```go
type UserProfile struct {
	Name     string  `json:"name" description:"User's full name" required:"true"`
	Email    string  `json:"email" description:"Email address" pattern:"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$" required:"true"`
	Age      int     `json:"age" description:"Age in years" min:"0" max:"150"`
	Role     string  `json:"role" description:"User role" enum:"admin,user,guest"`
	Username string  `json:"username" description:"Unique username" minLength:"3" maxLength:"20" pattern:"^[a-z0-9]+$" required:"true"`
	Bio      string  `json:"bio" description:"User biography" maxLength:"500"`
	Address  Address `json:"address" description:"User's address"`
	Tags     []string `json:"tags" description:"User tags" minItems:"1" maxItems:"10"`
}

type Address struct {
	Street  string `json:"street" required:"true"`
	City    string `json:"city" required:"true"`
	Country string `json:"country" required:"true"`
	ZipCode string `json:"zip_code" pattern:"^[0-9]{5}$"`
}

// Convert struct to Parameter
param, err := gollem.ToSchema(UserProfile{})
if err != nil {
	return err
}

// Set Title and Description
param.Title = "UserProfile"
param.Description = "Structured user profile information"

// Use with LLM
session, err := client.NewSession(ctx,
	gollem.WithSessionContentType(gollem.ContentTypeJSON),
	gollem.WithSessionResponseSchema(param),
)
```

### Supported Struct Tags

All tags are optional except `json` for field naming:

- **`json:"field_name"`** - Field name in JSON (standard Go tag)
  - Use `json:"-"` to ignore fields
- **`description:"text"`** - Field description
- **`enum:"value1,value2,value3"`** - Enum values (comma-separated)
- **`min:"0"`** - Minimum value (for numbers)
- **`max:"100"`** - Maximum value (for numbers)
- **`minLength:"1"`** - Minimum string length
- **`maxLength:"255"`** - Maximum string length
- **`pattern:"^[a-z]+$"`** - Regex pattern for string validation
- **`minItems:"1"`** - Minimum array length
- **`maxItems:"10"`** - Maximum array length
- **`required:"true"`** - Mark field as required

`ToSchema` checks the generated schema with `Parameter.Validate` before returning it. A tag combination that the schema cannot hold, such as `min` greater than `max`, `minLength` greater than `maxLength`, `minItems` greater than `maxItems`, or a `pattern` that does not compile, returns `gollem.ErrInvalidParameter`. The error message names the field, for example `invalid property "name": invalid pattern`.

### Supported Types

- **Basic types**: `string`, `int`, `int8`, `int16`, `int32`, `int64`, `uint`, `uint8`, `uint16`, `uint32`, `uint64`, `float32`, `float64`, `bool`
- **Complex types**: `[]T` (arrays/slices), nested structs, `map[K]V` (see [Maps](#maps))
- **Pointer types**: Automatically unwrapped to base type
- **Not supported**: interface types such as `any`, including `map[string]any`; channels and functions. These return `gollem.ErrUnsupportedType`.

### Maps

`ToSchema` converts a map to an object whose `AdditionalProperties` holds the schema of the values, for example `map[string]int` becomes `{"type": "object", "additionalProperties": {"type": "integer"}}`.

- The key type must be one that `encoding/json` can both encode and decode as an object key, because `Query[T]`, `SessionQuery[T]` and `NewTool` decode the LLM's JSON into your type: a string type, an integer type, or a type that implements `encoding.TextMarshaler` and whose pointer implements `encoding.TextUnmarshaler`. Other key types, including a type that implements only `MarshalText`, return `gollem.ErrUnsupportedType`.
- The schema does not restrict the keys, so the LLM may return a key that your key type cannot parse, such as a non-numeric key for `map[int]string`. Decoding then fails, and `Query[T]` asks the LLM to correct its response.
- The value type can be any supported type, including structs, slices and other maps. A map value must not be JSON `null`; validation rejects it, because the value schema does not allow null and decoding null would silently produce the zero value.

A map can be sent to an LLM in some ways but not in others:

| Where the schema is sent | Map |
|---|---|
| Tool definitions (Claude, OpenAI, Gemini) | Sent |
| Tool definitions (Ollama) | Sent, but the server ignores `additionalProperties` |
| Response schema, Ollama | Sent (as `format`) |
| Response schema, OpenAI without strict mode (the default) | Sent |
| Response schema, Gemini | Sent (as `responseJsonSchema`) |
| Response schema, Claude models or configurations without structured outputs (schema in the system prompt) | Sent |
| Response schema, Claude structured outputs | Rejected with `gollem.ErrUnsupportedSchema` |
| Response schema, OpenAI strict mode | Rejected with `gollem.ErrUnsupportedSchema` |

Claude structured outputs and OpenAI strict mode require `additionalProperties` to be `false` on every object, so they cannot express a map. gollem returns `gollem.ErrUnsupportedSchema` before calling the API, and the message names the map, for example `map at "labels" cannot be sent as Claude structured outputs`. The schema is sent unchanged in every other case; gollem does not rewrite a map into another shape. This applies equally to `Query[T]`, `SessionQuery[T]`, `WithSessionResponseSchema` and `WithGenerateResponseSchema`.

Gemini's `genai.Schema` has no `additionalProperties` field, so gollem sends a tool or response schema that contains a map as JSON Schema (`parametersJsonSchema` / `responseJsonSchema`), and every other schema in the `genai.Schema` form as before.

```go
type Result struct {
	Capitals map[string]string `json:"capitals" description:"capital city keyed by country name"`
}

resp, err := gollem.Query[Result](ctx, client, "Capitals of Japan and France")
if errors.Is(err, gollem.ErrUnsupportedSchema) {
	// The client sends response schemas in a way that cannot express a map,
	// e.g. Claude structured outputs.
}
```

### MustToSchema

For static initializations where errors should be caught at development time:

```go
var userSchema = gollem.MustToSchema(UserProfile{})
// Panics if conversion fails
```

### ToolSchema / MustToolSchema

`ToSchema` converts a single Go value into a schema. When defining a **tool**, use
`ToolSchema[In, Out]()` instead: it infers the input schema from the `In` type and
also verifies that the `Out` type can form a valid tool result (a JSON object).

```go
// Returns the input parameter schema; errors if In/Out are not valid tool types.
schema, err := gollem.ToolSchema[AddArgs, AddResult]()

// Panics on error — useful as a startup assertion of a tool's In/Out types.
schema := gollem.MustToolSchema[AddArgs, AddResult]()
```

These power [`NewTool` / `MustNewTool`](tools.md#type-safe-tools-with-newtool), which
build a `Tool` directly from typed handlers. `Out` has no slot in `ToolSpec`, so it is
validated (not returned) to ensure the handler's result is convertible to the wire form.

### Complete Example

See [examples/json_schema](../examples/json_schema) for a complete working example.

## Helper Functions

### Pointer Utility

Since many schema fields accept pointers, use this helper:

```go
func Ptr[T any](v T) *T {
	return &v
}

// Usage
schema := &gollem.Parameter{
	Type:    gollem.TypeInteger,
	Minimum: Ptr(0.0),
	Maximum: Ptr(100.0),
}
```

## Provider-Specific Behavior

### OpenAI

OpenAI uses Structured Outputs with JSON Schema:
- Supports all JSON Schema features
- gollem sends response schemas with `strict: false` by default. Strict mode requires `additionalProperties: false` on every object, so a schema with a map is rejected in strict mode (see [Maps](#maps)). Strict mode also rejects `AnyOf` on the root of the schema (see [Unions](#unions)).
- Compatible with GPT-4o and later models

### Claude

On models that support [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), Claude receives the schema as `output_config.format`:
- The API constrains decoding to the schema, so the response text is JSON that matches it. The documented exceptions are a refusal (`stop_reason: "refusal"`), a response cut off by `max_tokens`, and a string `enum` value returned with different capitalization, which the Claude documentation avoids by using lowercase enum values. As with other models, gollem extracts the JSON from the response text whenever the content type is JSON.
- gollem does not change the system prompt on these models, with or without a schema. With `ContentTypeJSON` and no schema, nothing is added, so ask for JSON in your own prompt. The system prompt and tool list are therefore the same as in a call without a schema; Claude accepts thinking blocks from earlier turns only while both stay the same, so a call with a schema can follow a tool loop over the same history.
- Structured outputs rejects some JSON Schema keywords. gollem removes `minimum`, `maximum`, `minLength`, `maxLength`, `pattern`, `maxItems` and a `minItems` above 1 from the schema and appends them to the field's description, for example `{maxLength: 20}`. The API does not enforce these constraints, and gollem does not validate the response against them. Every object is sent with `additionalProperties: false`. A schema with more than 16 parameters that use `AnyOf` is rejected with `gollem.ErrUnsupportedSchema` (see [Unions](#unions)).
- Models that do not support structured outputs (Claude Sonnet 4, Claude Opus 4 and the Claude 3 and Claude 2 models) behave as in earlier gollem versions: a JSON instruction and the schema are appended to the system prompt. The list is `structuredOutputsUnsupported` in `llm/claude/model.go`; any other model ID, including one served through `WithBaseURL`, is sent `output_config.format`.
- On Vertex AI, a Google Cloud organization policy can forbid structured outputs (`constraints/vertexai.allowedPartnerModelFeatures` without `structured_outputs` for the model), and Vertex AI then rejects the call with a 400. Create the client with `claude.WithVertexStructuredOutputsDisabled()` to send every model the schema in the system prompt instead. The system prompt then changes with the schema, so a history containing thinking blocks produced under another system prompt is rejected.

### Gemini

Gemini uses `response_schema` parameter:
- Native JSON Schema support via Vertex AI
- Works with Gemini 1.5 Pro and later
- Automatically validates response format

## Best Practices

### 1. Provide Clear Descriptions

```go
&gollem.Parameter{
	Type:        gollem.TypeString,
	Description: "User's full legal name as it appears on government ID",
	// Better than: Description: "name"
}
```

### 2. Use Appropriate Constraints

```go
// Email validation
&gollem.Parameter{
	Type:    gollem.TypeString,
	Pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
}

// Age constraint
&gollem.Parameter{
	Type:    gollem.TypeInteger,
	Minimum: Ptr(0.0),
	Maximum: Ptr(150.0),
}
```

### 3. Mark Required Fields

```go
&gollem.Parameter{
	Type: gollem.TypeObject,
	Properties: map[string]*gollem.Parameter{
		"required_field": {Type: gollem.TypeString},
		"optional_field": {Type: gollem.TypeString},
	},
	Required: []string{"required_field"},
}
```

### 4. Use Enums for Fixed Values

```go
&gollem.Parameter{
	Type: gollem.TypeString,
	Enum: []any{"pending", "approved", "rejected"},
	Description: "Application status - must be one of the allowed values",
}
```

## Common Use Cases

### Data Extraction

Extract structured information from unstructured text:

```go
schema := &gollem.ResponseSchema{
	Name: "InvoiceData",
	Schema: &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"invoice_number": {Type: gollem.TypeString},
			"date":           {Type: gollem.TypeString},
			"total_amount":   {Type: gollem.TypeNumber},
			"items": {
				Type: gollem.TypeArray,
				Items: &gollem.Parameter{
					Type: gollem.TypeObject,
					Properties: map[string]*gollem.Parameter{
						"description": {Type: gollem.TypeString},
						"quantity":    {Type: gollem.TypeInteger},
						"price":       {Type: gollem.TypeNumber},
					},
				},
			},
		},
	},
}
```

### Classification

Classify content into predefined categories:

```go
schema := &gollem.ResponseSchema{
	Name: "ContentClassification",
	Schema: &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"category": {
				Type: gollem.TypeString,
				Enum: []any{"technical", "business", "personal", "other"},
			},
			"subcategory": {Type: gollem.TypeString},
			"confidence": {
				Type:    gollem.TypeNumber,
				Minimum: Ptr(0.0),
				Maximum: Ptr(1.0),
			},
		},
		Required: []string{"category", "confidence"},
	},
}
```

### Form Generation

Generate structured forms or questionnaires:

```go
schema := &gollem.ResponseSchema{
	Name: "SurveyQuestions",
	Schema: &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"questions": {
				Type: gollem.TypeArray,
				Items: &gollem.Parameter{
					Type: gollem.TypeObject,
					Properties: map[string]*gollem.Parameter{
						"question": {Type: gollem.TypeString},
						"type": {
							Type: gollem.TypeString,
							Enum: []any{"multiple_choice", "text", "rating"},
						},
						"options": {
							Type:  gollem.TypeArray,
							Items: &gollem.Parameter{Type: gollem.TypeString},
						},
					},
					Required: []string{"question", "type"},
				},
				MinItems: Ptr(3),
				MaxItems: Ptr(10),
			},
		},
	},
}
```

## Error Handling

When schema validation fails (rare with proper setup):

```go
resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("...")})
if err != nil {
	return fmt.Errorf("failed to generate content: %w", err)
}

var result map[string]any
if err := json.Unmarshal([]byte(resp.Texts[0]), &result); err != nil {
	return fmt.Errorf("failed to parse JSON response: %w", err)
}
```

## Complete Example

See [examples/json_schema](../examples/json_schema) for a complete working example demonstrating:
- Schema definition for user profiles
- Extraction of structured data from natural language
- Usage with OpenAI, Claude, and Gemini
- Pretty-printing JSON output

## Related Documentation

- [Getting Started Guide](getting-started.md)
- [Tool Development](tools.md)
- [LLM Provider Configuration](llm.md)
