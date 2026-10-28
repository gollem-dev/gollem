package gollem_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/gollem-dev/gollem/llm/gemini"
	"github.com/gollem-dev/gollem/llm/openai"
	"github.com/gollem-dev/gollem/mock"
	"github.com/m-mizutani/gt"
)

// addArgs/addResult are typed tool definitions used across the typed-tool tests.
// "a" is required and "b" is optional, so required/optional propagation is testable.
type addArgs struct {
	A float64 `json:"a" description:"First number" required:"true"`
	B float64 `json:"b" description:"Second number"`
}

type addResult struct {
	Result float64 `json:"result"`
}

func addHandler(_ context.Context, in addArgs) (addResult, error) {
	return addResult{Result: in.A + in.B}, nil
}

// badTagArgs has an invalid struct tag so that schema inference fails.
type badTagArgs struct {
	A int `json:"a" min:"not-a-number"`
}

// richArgs exercises the full set of supported constraint tags, including a
// nested struct and an array, to confirm they all propagate through ToolSchema.
type richArgs struct {
	Name string   `json:"name" description:"the name" required:"true" minLength:"1" maxLength:"10" pattern:"^[a-z]+$"`
	Age  int      `json:"age" min:"0" max:"150"`
	Role string   `json:"role" enum:"admin,user,guest"`
	Tags []string `json:"tags" minItems:"1" maxItems:"5"`
	Addr addr     `json:"addr"`
}

type addr struct {
	City string `json:"city" required:"true"`
}

func mustPanic(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic, but the function did not panic")
		}
	}()
	fn()
}

func TestToolSchema(t *testing.T) {
	t.Run("struct In/Out succeeds and reflects tags", func(t *testing.T) {
		schema, err := gollem.ToolSchema[addArgs, addResult]()
		gt.NoError(t, err)
		gt.Equal(t, gollem.TypeObject, schema.Type)

		a := schema.Properties["a"]
		gt.NotNil(t, a)
		gt.Equal(t, gollem.TypeNumber, a.Type)
		gt.True(t, a.Required)
		gt.Equal(t, "First number", a.Description)

		b := schema.Properties["b"]
		gt.NotNil(t, b)
		gt.Equal(t, gollem.TypeNumber, b.Type)
		gt.False(t, b.Required)
	})

	t.Run("all constraint tags propagate through ToolSchema", func(t *testing.T) {
		schema, err := gollem.ToolSchema[richArgs, addResult]()
		gt.NoError(t, err)
		p := schema.Properties

		// string: description, required, length, pattern
		gt.Equal(t, "the name", p["name"].Description)
		gt.True(t, p["name"].Required)
		gt.Equal(t, 1, *p["name"].MinLength)
		gt.Equal(t, 10, *p["name"].MaxLength)
		gt.Equal(t, "^[a-z]+$", p["name"].Pattern)

		// integer: min/max
		gt.Equal(t, float64(0), *p["age"].Minimum)
		gt.Equal(t, float64(150), *p["age"].Maximum)

		// enum
		gt.Equal(t, []string{"admin", "user", "guest"}, p["role"].Enum)

		// array: item type + size constraints
		gt.Equal(t, gollem.TypeArray, p["tags"].Type)
		gt.Equal(t, gollem.TypeString, p["tags"].Items.Type)
		gt.Equal(t, 1, *p["tags"].MinItems)
		gt.Equal(t, 5, *p["tags"].MaxItems)

		// nested struct: recursive properties + required
		gt.Equal(t, gollem.TypeObject, p["addr"].Type)
		gt.True(t, p["addr"].Properties["city"].Required)
	})

	t.Run("map[string]any Out is accepted", func(t *testing.T) {
		schema, err := gollem.ToolSchema[addArgs, map[string]any]()
		gt.NoError(t, err)
		gt.Equal(t, gollem.TypeObject, schema.Type)
	})

	t.Run("scalar In is rejected", func(t *testing.T) {
		_, err := gollem.ToolSchema[string, addResult]()
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrInvalidToolType))
	})

	t.Run("map In is rejected", func(t *testing.T) {
		// A map In has no named parameters to declare, so it is not a valid input type.
		_, err := gollem.ToolSchema[map[string]any, addResult]()
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrInvalidToolType))
	})

	t.Run("scalar Out is rejected", func(t *testing.T) {
		_, err := gollem.ToolSchema[addArgs, string]()
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrInvalidToolType))
	})

	t.Run("interface Out is rejected", func(t *testing.T) {
		// any has a nil zero value; a zero-value probe would miss it, but the
		// reflect-based check rejects it deterministically.
		_, err := gollem.ToolSchema[addArgs, any]()
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrInvalidToolType))
	})

	t.Run("invalid struct tag is rejected", func(t *testing.T) {
		_, err := gollem.ToolSchema[badTagArgs, addResult]()
		gt.Error(t, err)
	})
}

func TestMustToolSchema(t *testing.T) {
	t.Run("returns schema for valid types", func(t *testing.T) {
		schema := gollem.MustToolSchema[addArgs, addResult]()
		gt.NotNil(t, schema)
		gt.Equal(t, gollem.TypeObject, schema.Type)
	})

	t.Run("panics for invalid types", func(t *testing.T) {
		mustPanic(t, func() {
			_ = gollem.MustToolSchema[string, addResult]()
		})
	})
}

func TestNewTool(t *testing.T) {
	ctx := t.Context()

	t.Run("spec reflects name, description and inferred parameters", func(t *testing.T) {
		tool, err := gollem.NewTool("add", "Adds two numbers", addHandler)
		gt.NoError(t, err)

		spec := tool.Spec()
		gt.Equal(t, "add", spec.Name)
		gt.Equal(t, "Adds two numbers", spec.Description)
		gt.Equal(t, gollem.TypeNumber, spec.Parameters["a"].Type)
		gt.True(t, spec.Parameters["a"].Required)
	})

	t.Run("run decodes args and converts struct result", func(t *testing.T) {
		var got addArgs
		tool, err := gollem.NewTool("add", "Adds two numbers",
			func(_ context.Context, in addArgs) (addResult, error) {
				got = in
				return addResult{Result: in.A + in.B}, nil
			})
		gt.NoError(t, err)

		out, err := tool.Run(ctx, map[string]any{"a": float64(2), "b": float64(3)})
		gt.NoError(t, err)
		gt.Equal(t, float64(2), got.A)
		gt.Equal(t, float64(3), got.B)
		gt.Equal(t, float64(5), out["result"].(float64))
	})

	t.Run("map result is passed through", func(t *testing.T) {
		tool, err := gollem.NewTool("echo", "Echoes input",
			func(_ context.Context, in addArgs) (map[string]any, error) {
				return map[string]any{"sum": in.A + in.B}, nil
			})
		gt.NoError(t, err)

		out, err := tool.Run(ctx, map[string]any{"a": float64(4), "b": float64(1)})
		gt.NoError(t, err)
		gt.Equal(t, float64(5), out["sum"].(float64))
	})

	t.Run("decode failure is propagated", func(t *testing.T) {
		tool, err := gollem.NewTool("add", "Adds two numbers", addHandler)
		gt.NoError(t, err)

		_, err = tool.Run(ctx, map[string]any{"a": "not-a-number"})
		gt.Error(t, err)
	})

	t.Run("handler error is propagated", func(t *testing.T) {
		sentinel := errors.New("boom")
		tool, err := gollem.NewTool("fail", "Always fails",
			func(_ context.Context, _ addArgs) (addResult, error) {
				return addResult{}, sentinel
			})
		gt.NoError(t, err)

		_, err = tool.Run(ctx, map[string]any{"a": float64(1), "b": float64(2)})
		gt.True(t, errors.Is(err, sentinel))
	})

	t.Run("invalid In type returns error", func(t *testing.T) {
		_, err := gollem.NewTool("bad", "bad",
			func(_ context.Context, _ string) (addResult, error) {
				return addResult{}, nil
			})
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrInvalidToolType))
	})

	t.Run("empty name returns error", func(t *testing.T) {
		_, err := gollem.NewTool("", "no name", addHandler)
		gt.Error(t, err)
	})
}

func TestNewToolRejectsConstraintViolations(t *testing.T) {
	// Proves that a NewTool-produced spec enforces every constraint kind, not just
	// types: the inferred schema, run through the standard ValidateArgs, rejects
	// violating input. (ValidateArgs is what executeToolCall runs before the handler.)
	tool, err := gollem.NewTool("rich", "rich tool",
		func(_ context.Context, _ richArgs) (addResult, error) {
			return addResult{}, nil
		})
	gt.NoError(t, err)
	spec := tool.Spec()

	validArgs := func() map[string]any {
		return map[string]any{
			"name": "abc",
			"age":  float64(30),
			"role": "admin",
			"tags": []any{"x"},
			"addr": map[string]any{"city": "Tokyo"},
		}
	}

	t.Run("valid args pass", func(t *testing.T) {
		gt.NoError(t, spec.ValidateArgs(validArgs()))
	})

	reject := func(mutate func(m map[string]any)) func(t *testing.T) {
		return func(t *testing.T) {
			m := validArgs()
			mutate(m)
			err := spec.ValidateArgs(m)
			gt.Error(t, err)
			gt.True(t, errors.Is(err, gollem.ErrToolArgsValidation))
		}
	}

	t.Run("missing required", reject(func(m map[string]any) { delete(m, "name") }))
	t.Run("enum violation", reject(func(m map[string]any) { m["role"] = "root" }))
	t.Run("below minimum", reject(func(m map[string]any) { m["age"] = float64(-1) }))
	t.Run("above maximum", reject(func(m map[string]any) { m["age"] = float64(200) }))
	t.Run("too short", reject(func(m map[string]any) { m["name"] = "" }))
	t.Run("pattern mismatch", reject(func(m map[string]any) { m["name"] = "ABC" }))
	t.Run("too few items", reject(func(m map[string]any) { m["tags"] = []any{} }))
	t.Run("nested required missing", reject(func(m map[string]any) { m["addr"] = map[string]any{} }))
}

func TestMustNewTool(t *testing.T) {
	t.Run("returns tool for valid definition", func(t *testing.T) {
		tool := gollem.MustNewTool("add", "Adds two numbers", addHandler)
		gt.NotNil(t, tool)
		gt.Equal(t, "add", tool.Spec().Name)
	})

	t.Run("panics for invalid definition", func(t *testing.T) {
		mustPanic(t, func() {
			_ = gollem.MustNewTool("bad", "bad",
				func(_ context.Context, _ string) (addResult, error) {
					return addResult{}, nil
				})
		})
	})
}

func TestNewToolWithMockLLM(t *testing.T) {
	t.Run("definition propagates and handler receives typed input", func(t *testing.T) {
		var gotIn addArgs
		handlerCalled := false
		tool, err := gollem.NewTool("add", "Adds two numbers",
			func(_ context.Context, in addArgs) (addResult, error) {
				gotIn = in
				handlerCalled = true
				return addResult{Result: in.A + in.B}, nil
			})
		gt.NoError(t, err)

		var propagated gollem.ToolSpec
		specCaptured := false
		callCount := 0
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(_ context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				// The agent passes tools to the session via WithSessionTools; inspect
				// what is actually handed to the LLM to confirm the inferred schema.
				cfg := gollem.NewSessionConfig(options...)
				for _, tl := range cfg.Tools() {
					if tl.Spec().Name == "add" {
						propagated = tl.Spec()
						specCaptured = true
					}
				}

				return &mock.SessionMock{
					GenerateFunc: func(_ context.Context, _ []gollem.Input, _ ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++
						if callCount == 1 {
							return &gollem.Response{
								FunctionCalls: []*gollem.FunctionCall{
									{
										ID:        "call_add_1",
										Name:      "add",
										Arguments: map[string]any{"a": float64(2), "b": float64(3)},
									},
								},
							}, nil
						}
						return &gollem.Response{Texts: []string{"done"}}, nil
					},
				}, nil
			},
		}

		agent := gollem.New(mockClient, gollem.WithTools(tool), gollem.WithLoopLimit(5))
		_, err = agent.Execute(t.Context(), gollem.Text("add 2 and 3"))
		gt.NoError(t, err)

		// Tool definition propagated to the LLM exactly as inferred from addArgs.
		gt.True(t, specCaptured)
		gt.Equal(t, "add", propagated.Name)
		gt.Equal(t, "Adds two numbers", propagated.Description)
		gt.Equal(t, gollem.TypeNumber, propagated.Parameters["a"].Type)
		gt.True(t, propagated.Parameters["a"].Required)
		gt.Equal(t, "First number", propagated.Parameters["a"].Description)
		gt.False(t, propagated.Parameters["b"].Required)

		// Handler received decoded, correctly typed values.
		gt.True(t, handlerCalled)
		gt.Equal(t, float64(2), gotIn.A)
		gt.Equal(t, float64(3), gotIn.B)
	})

	t.Run("invalid args are blocked before the handler runs", func(t *testing.T) {
		handlerCalled := false
		tool, err := gollem.NewTool("add", "Adds two numbers",
			func(_ context.Context, in addArgs) (addResult, error) {
				handlerCalled = true
				return addResult{Result: in.A + in.B}, nil
			})
		gt.NoError(t, err)

		callCount := 0
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(_ context.Context, _ ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(_ context.Context, _ []gollem.Input, _ ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++
						if callCount == 1 {
							// "a" is required but the wrong type; ValidateArgs must reject it.
							return &gollem.Response{
								FunctionCalls: []*gollem.FunctionCall{
									{ID: "call_add_1", Name: "add", Arguments: map[string]any{"a": "not-a-number"}},
								},
							}, nil
						}
						return &gollem.Response{Texts: []string{"done"}}, nil
					},
				}, nil
			},
		}

		agent := gollem.New(mockClient, gollem.WithTools(tool), gollem.WithLoopLimit(5))
		_, err = agent.Execute(t.Context(), gollem.Text("add"))
		gt.NoError(t, err)
		gt.False(t, handlerCalled)
	})

	t.Run("with validation disabled, missing required reaches handler as zero value", func(t *testing.T) {
		// Pins the documented behavior: WithDisableArgsValidation skips ValidateArgs,
		// and decode does not enforce required, so a missing field arrives as its zero value.
		var gotIn addArgs
		handlerCalled := false
		tool, err := gollem.NewTool("add", "Adds two numbers",
			func(_ context.Context, in addArgs) (addResult, error) {
				gotIn = in
				handlerCalled = true
				return addResult{Result: in.A + in.B}, nil
			})
		gt.NoError(t, err)

		callCount := 0
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(_ context.Context, _ ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(_ context.Context, _ []gollem.Input, _ ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++
						if callCount == 1 {
							// "a" (required) omitted entirely.
							return &gollem.Response{
								FunctionCalls: []*gollem.FunctionCall{
									{ID: "call_add_1", Name: "add", Arguments: map[string]any{"b": float64(3)}},
								},
							}, nil
						}
						return &gollem.Response{Texts: []string{"done"}}, nil
					},
				}, nil
			},
		}

		agent := gollem.New(mockClient,
			gollem.WithTools(tool),
			gollem.WithLoopLimit(5),
			gollem.WithDisableArgsValidation(),
		)
		_, err = agent.Execute(t.Context(), gollem.Text("add"))
		gt.NoError(t, err)
		gt.True(t, handlerCalled)
		gt.Equal(t, float64(0), gotIn.A)
		gt.Equal(t, float64(3), gotIn.B)
	})
}

// encoding/json decodes a key into a struct field without regard to letter case,
// so a typed tool must not decode a key that differs from a schema name only in
// letter case: its value would reach the handler without being validated.
func TestTypedToolRejectsCaseVariantKeys(t *testing.T) {
	type grant struct {
		Role string `json:"role" enum:"user,guest"`
	}
	type setRoleArgs struct {
		Role   string  `json:"role" enum:"user,guest"`
		Grants []grant `json:"grants"`
	}

	newSetRole := func(t *testing.T, called *bool) gollem.Tool {
		t.Helper()
		tool, err := gollem.NewTool("set_role", "Sets the role",
			func(_ context.Context, _ setRoleArgs) (addResult, error) {
				*called = true
				return addResult{}, nil
			})
		gt.NoError(t, err)
		return tool
	}

	t.Run("Run rejects a capitalized key without calling the handler", func(t *testing.T) {
		called := false
		_, err := newSetRole(t, &called).Run(t.Context(), map[string]any{"Role": "admin"})
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrToolArgsValidation))
		gt.S(t, err.Error()).Contains(`key "Role" differs from declared parameter "role"`)
		gt.False(t, called)
	})

	t.Run("Run rejects a case variant inside an array of structs", func(t *testing.T) {
		called := false
		_, err := newSetRole(t, &called).Run(t.Context(), map[string]any{
			"grants": []any{map[string]any{"Role": "admin"}},
		})
		gt.Error(t, err)
		gt.S(t, err.Error()).Contains(`key "grants[0].Role" differs from declared parameter "role"`)
		gt.False(t, called)
	})

	t.Run("Run rejects a case variant held in a typed Go slice of maps", func(t *testing.T) {
		// A caller running the tool directly can pass Go-typed values that
		// json.Marshal encodes as objects; the check must see them too.
		called := false
		_, err := newSetRole(t, &called).Run(t.Context(), map[string]any{
			"grants": []map[string]string{{"Role": "admin"}},
		})
		gt.Error(t, err)
		gt.S(t, err.Error()).Contains(`key "grants[0].Role" differs from declared parameter "role"`)
		gt.False(t, called)
	})

	t.Run("Run accepts declared keys", func(t *testing.T) {
		called := false
		_, err := newSetRole(t, &called).Run(t.Context(), map[string]any{
			"role":   "user",
			"grants": []any{map[string]any{"role": "guest"}},
		})
		gt.NoError(t, err)
		gt.True(t, called)
	})

	runAgent := func(t *testing.T, args map[string]any, options ...gollem.Option) bool {
		t.Helper()
		called := false
		tool := newSetRole(t, &called)

		callCount := 0
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(_ context.Context, _ ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(_ context.Context, _ []gollem.Input, _ ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++
						if callCount == 1 {
							return &gollem.Response{
								FunctionCalls: []*gollem.FunctionCall{
									{ID: "call_set_role_1", Name: "set_role", Arguments: args},
								},
							}, nil
						}
						return &gollem.Response{Texts: []string{"done"}}, nil
					},
				}, nil
			},
		}

		options = append([]gollem.Option{gollem.WithTools(tool), gollem.WithLoopLimit(5)}, options...)
		agent := gollem.New(mockClient, options...)
		_, err := agent.Execute(t.Context(), gollem.Text("set role"))
		gt.NoError(t, err)
		return called
	}

	t.Run("agent blocks a capitalized key before the handler runs", func(t *testing.T) {
		gt.False(t, runAgent(t, map[string]any{"Role": "admin"}))
	})

	t.Run("agent blocks a capitalized key with validation disabled", func(t *testing.T) {
		gt.False(t, runAgent(t, map[string]any{"Role": "admin"}, gollem.WithDisableArgsValidation()))
	})
}

// A typed tool returning an int64 wider than float64 must reach the LLM with the value it
// returned. toResultMap used to decode through a plain map[string]any, which rounded it.
func TestTypedToolResultPreservesWideIntegers(t *testing.T) {
	type lookupArgs struct {
		Key string `json:"key" description:"Lookup key" required:"true"`
	}
	type lookupResult struct {
		AccountID int64 `json:"account_id" description:"Account identifier"`
	}

	tool, err := gollem.NewTool("lookup", "Looks an account up",
		func(_ context.Context, _ lookupArgs) (lookupResult, error) {
			return lookupResult{AccountID: 9007199254740993}, nil
		})
	gt.NoError(t, err)

	result, err := tool.Run(t.Context(), map[string]any{"key": "k"})
	gt.NoError(t, err)

	encoded, err := json.Marshal(result)
	gt.NoError(t, err)
	gt.Equal(t, `{"account_id":9007199254740993}`, string(encoded))
}

type setLabelsArgs struct {
	ID     string            `json:"id" description:"Resource ID" required:"true"`
	Labels map[string]string `json:"labels" description:"Labels to set, keyed by label name" required:"true"`
}

type setLabelsResult struct {
	Count int `json:"count"`
}

func newSetLabelsTool(t *testing.T, received *map[string]string) gollem.Tool {
	t.Helper()
	tool, err := gollem.NewTool("set_labels", "Sets labels on a resource",
		func(_ context.Context, in setLabelsArgs) (setLabelsResult, error) {
			*received = in.Labels
			return setLabelsResult{Count: len(in.Labels)}, nil
		})
	gt.NoError(t, err).Required()
	return tool
}

func TestNewToolWithMapArgument(t *testing.T) {
	var received map[string]string
	tool := newSetLabelsTool(t, &received)

	labels := tool.Spec().Parameters["labels"]
	gt.Equal(t, gollem.TypeObject, labels.Type)
	gt.Equal(t, gollem.TypeString, labels.AdditionalProperties.Type)

	t.Run("valid map argument is decoded", func(t *testing.T) {
		args := map[string]any{"id": "r1", "labels": map[string]any{"env": "prod", "team": "core"}}
		spec := tool.Spec()
		gt.NoError(t, spec.ValidateArgs(args))
		out, err := tool.Run(t.Context(), args)
		gt.NoError(t, err)
		gt.Equal(t, map[string]string{"env": "prod", "team": "core"}, received)
		encoded, err := json.Marshal(out)
		gt.NoError(t, err)
		gt.Equal(t, `{"count":2}`, string(encoded))
	})

	t.Run("map value of wrong type is rejected", func(t *testing.T) {
		spec := tool.Spec()
		err := spec.ValidateArgs(map[string]any{"id": "r1", "labels": map[string]any{"env": float64(1)}})
		gt.True(t, errors.Is(err, gollem.ErrToolArgsValidation))
		gt.S(t, err.Error()).Contains("expected string type")
	})
}

func TestNewToolRejectsInvalidTagCombination(t *testing.T) {
	type patternArgs struct {
		Name string `json:"name" pattern:"[a-"`
	}
	_, err := gollem.NewTool("bad", "bad pattern",
		func(_ context.Context, _ patternArgs) (setLabelsResult, error) { return setLabelsResult{}, nil })
	gt.True(t, errors.Is(err, gollem.ErrInvalidParameter))
	gt.S(t, err.Error()).Contains(`invalid property "name": invalid pattern`)
}

// TestNewToolWithMapArgumentIntegration checks against the real APIs that a
// tool definition with a map argument is accepted and that the map the LLM
// sends is decoded into the typed argument.
func TestNewToolWithMapArgumentIntegration(t *testing.T) {
	t.Parallel()

	testFn := func(t *testing.T, client gollem.LLMClient) {
		ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
		defer cancel()

		var received map[string]string
		tool := newSetLabelsTool(t, &received)

		session, err := client.NewSession(ctx, gollem.WithSessionTools(tool))
		gt.NoError(t, err).Required()

		resp, err := session.Generate(ctx, []gollem.Input{gollem.Text(
			`Call set_labels for resource "r1" with the labels env=prod and team=core.`)})
		gt.NoError(t, err).Required()
		gt.A(t, resp.FunctionCalls).Longer(0).Required()

		call := resp.FunctionCalls[0]
		gt.Equal(t, "set_labels", call.Name)
		spec := tool.Spec()
		gt.NoError(t, spec.ValidateArgs(call.Arguments))
		_, err = tool.Run(ctx, call.Arguments)
		gt.NoError(t, err)
		t.Logf("labels: %v", received)
		gt.Equal(t, "prod", received["env"])
		gt.Equal(t, "core", received["team"])
	}

	t.Run("OpenAI", func(t *testing.T) {
		t.Parallel()
		apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
		if !ok {
			t.Skip("TEST_OPENAI_API_KEY is not set")
		}
		client, err := openai.New(context.Background(), apiKey)
		gt.NoError(t, err).Required()
		testFn(t, client)
	})

	t.Run("Claude", func(t *testing.T) {
		t.Parallel()
		apiKey, ok := os.LookupEnv("TEST_CLAUDE_API_KEY")
		if !ok {
			t.Skip("TEST_CLAUDE_API_KEY is not set")
		}
		client, err := claude.New(context.Background(), apiKey)
		gt.NoError(t, err).Required()
		testFn(t, client)
	})

	t.Run("Gemini", func(t *testing.T) {
		t.Parallel()
		projectID, ok := os.LookupEnv("TEST_GCP_PROJECT_ID")
		if !ok {
			t.Skip("TEST_GCP_PROJECT_ID is not set")
		}
		location, ok := os.LookupEnv("TEST_GCP_LOCATION")
		if !ok {
			t.Skip("TEST_GCP_LOCATION is not set")
		}
		var opts []gemini.Option
		if model := os.Getenv("TEST_GCP_MODEL"); model != "" {
			opts = append(opts, gemini.WithModel(model))
		}
		client, err := gemini.New(context.Background(), projectID, location, opts...)
		gt.NoError(t, err).Required()
		testFn(t, client)
	})
}
