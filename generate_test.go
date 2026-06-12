package gollem_test

import (
	"context"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/mock"
	"github.com/m-mizutani/gt"
)

func TestGenerateConfigDefaults(t *testing.T) {
	cfg := gollem.NewGenerateConfig()

	gt.Value(t, cfg.ResponseSchema()).Equal((*gollem.Parameter)(nil))
	gt.Value(t, cfg.Temperature()).Equal((*float64)(nil))
	gt.Value(t, cfg.TopP()).Equal((*float64)(nil))
	gt.Value(t, cfg.MaxTokens()).Equal((*int)(nil))
}

func TestGenerateConfigWithTemperature(t *testing.T) {
	cfg := gollem.NewGenerateConfig(gollem.WithTemperature(0.5))

	gt.NotNil(t, cfg.Temperature())
	gt.Value(t, *cfg.Temperature()).Equal(0.5)
	// Other fields remain nil
	gt.Value(t, cfg.TopP()).Equal((*float64)(nil))
	gt.Value(t, cfg.MaxTokens()).Equal((*int)(nil))
	gt.Value(t, cfg.ResponseSchema()).Equal((*gollem.Parameter)(nil))
}

func TestGenerateConfigWithTopP(t *testing.T) {
	cfg := gollem.NewGenerateConfig(gollem.WithTopP(0.9))

	gt.NotNil(t, cfg.TopP())
	gt.Value(t, *cfg.TopP()).Equal(0.9)
	gt.Value(t, cfg.Temperature()).Equal((*float64)(nil))
}

func TestGenerateConfigWithMaxTokens(t *testing.T) {
	cfg := gollem.NewGenerateConfig(gollem.WithMaxTokens(1024))

	gt.NotNil(t, cfg.MaxTokens())
	gt.Value(t, *cfg.MaxTokens()).Equal(1024)
	gt.Value(t, cfg.Temperature()).Equal((*float64)(nil))
}

func TestGenerateConfigWithResponseSchema(t *testing.T) {
	schema := &gollem.Parameter{
		Type:  gollem.TypeObject,
		Title: "TestSchema",
		Properties: map[string]*gollem.Parameter{
			"name": {Type: gollem.TypeString, Description: "name field"},
		},
	}
	cfg := gollem.NewGenerateConfig(gollem.WithGenerateResponseSchema(schema))

	gt.NotNil(t, cfg.ResponseSchema())
	gt.Value(t, cfg.ResponseSchema().Title).Equal("TestSchema")
	gt.Value(t, cfg.Temperature()).Equal((*float64)(nil))
}

func TestGenerateConfigMultipleOptions(t *testing.T) {
	schema := &gollem.Parameter{
		Type:  gollem.TypeObject,
		Title: "Multi",
	}
	cfg := gollem.NewGenerateConfig(
		gollem.WithTemperature(0.3),
		gollem.WithTopP(0.8),
		gollem.WithMaxTokens(512),
		gollem.WithGenerateResponseSchema(schema),
	)

	gt.NotNil(t, cfg.Temperature())
	gt.Value(t, *cfg.Temperature()).Equal(0.3)
	gt.NotNil(t, cfg.TopP())
	gt.Value(t, *cfg.TopP()).Equal(0.8)
	gt.NotNil(t, cfg.MaxTokens())
	gt.Value(t, *cfg.MaxTokens()).Equal(512)
	gt.NotNil(t, cfg.ResponseSchema())
	gt.Value(t, cfg.ResponseSchema().Title).Equal("Multi")
}

func TestGenerateConfigLastOptionWins(t *testing.T) {
	cfg := gollem.NewGenerateConfig(
		gollem.WithTemperature(0.1),
		gollem.WithTemperature(0.9),
	)

	gt.NotNil(t, cfg.Temperature())
	gt.Value(t, *cfg.Temperature()).Equal(0.9)
}

func TestGenerateConfigZeroValuesAreDistinctFromNil(t *testing.T) {
	cfg := gollem.NewGenerateConfig(
		gollem.WithTemperature(0.0),
		gollem.WithMaxTokens(0),
	)

	// 0.0 and 0 are valid values, distinct from nil (unset)
	gt.NotNil(t, cfg.Temperature())
	gt.Value(t, *cfg.Temperature()).Equal(0.0)
	gt.NotNil(t, cfg.MaxTokens())
	gt.Value(t, *cfg.MaxTokens()).Equal(0)
}

// --- Tests verifying Generate/Stream signature migration works correctly ---

func TestGenerateAcceptsSliceInput(t *testing.T) {
	// Verify that Generate correctly receives []Input (not variadic)
	// and passes it through to the implementation.

	var receivedInput []gollem.Input
	session := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			receivedInput = input
			return &gollem.Response{Texts: []string{"ok"}}, nil
		},
	}

	t.Run("single text input", func(t *testing.T) {
		input := []gollem.Input{gollem.Text("hello")}
		_, err := session.Generate(context.Background(), input)
		gt.NoError(t, err)
		gt.Value(t, len(receivedInput)).Equal(1)
		gt.Value(t, string(receivedInput[0].(gollem.Text))).Equal("hello")
	})

	t.Run("multiple inputs", func(t *testing.T) {
		input := []gollem.Input{
			gollem.Text("first"),
			gollem.Text("second"),
		}
		_, err := session.Generate(context.Background(), input)
		gt.NoError(t, err)
		gt.Value(t, len(receivedInput)).Equal(2)
		gt.Value(t, string(receivedInput[0].(gollem.Text))).Equal("first")
		gt.Value(t, string(receivedInput[1].(gollem.Text))).Equal("second")
	})

	t.Run("empty input slice", func(t *testing.T) {
		_, err := session.Generate(context.Background(), []gollem.Input{})
		gt.NoError(t, err)
		gt.Value(t, len(receivedInput)).Equal(0)
	})

	t.Run("nil input slice", func(t *testing.T) {
		_, err := session.Generate(context.Background(), nil)
		gt.NoError(t, err)
		gt.Value(t, receivedInput).Equal([]gollem.Input(nil))
	})
}

func TestGeneratePassesOptionsToProvider(t *testing.T) {
	// Verify that per-call GenerateOption values are correctly
	// forwarded from the caller to the provider's Generate method.

	var receivedOpts []gollem.GenerateOption
	session := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			receivedOpts = opts
			return &gollem.Response{Texts: []string{"ok"}}, nil
		},
	}

	t.Run("no options", func(t *testing.T) {
		_, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("test")})
		gt.NoError(t, err)
		gt.Value(t, len(receivedOpts)).Equal(0)
	})

	t.Run("with temperature", func(t *testing.T) {
		_, err := session.Generate(context.Background(),
			[]gollem.Input{gollem.Text("test")},
			gollem.WithTemperature(0.7),
		)
		gt.NoError(t, err)
		gt.Value(t, len(receivedOpts)).Equal(1)

		cfg := gollem.NewGenerateConfig(receivedOpts...)
		gt.NotNil(t, cfg.Temperature())
		gt.Value(t, *cfg.Temperature()).Equal(0.7)
	})

	t.Run("with multiple options", func(t *testing.T) {
		schema := &gollem.Parameter{Type: gollem.TypeObject, Title: "Test"}
		_, err := session.Generate(context.Background(),
			[]gollem.Input{gollem.Text("test")},
			gollem.WithTemperature(0.5),
			gollem.WithMaxTokens(100),
			gollem.WithGenerateResponseSchema(schema),
		)
		gt.NoError(t, err)
		gt.Value(t, len(receivedOpts)).Equal(3)

		cfg := gollem.NewGenerateConfig(receivedOpts...)
		gt.Value(t, *cfg.Temperature()).Equal(0.5)
		gt.Value(t, *cfg.MaxTokens()).Equal(100)
		gt.Value(t, cfg.ResponseSchema().Title).Equal("Test")
	})
}

func TestStreamAcceptsSliceInputAndOptions(t *testing.T) {
	// Verify that Stream correctly receives []Input and ...GenerateOption.

	var receivedInput []gollem.Input
	var receivedOpts []gollem.GenerateOption

	ch := make(chan *gollem.Response, 1)
	ch <- &gollem.Response{Texts: []string{"streamed"}}
	close(ch)

	session := &mock.SessionMock{
		StreamFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (<-chan *gollem.Response, error) {
			receivedInput = input
			receivedOpts = opts
			return ch, nil
		},
	}

	input := []gollem.Input{gollem.Text("stream me")}
	resultCh, err := session.Stream(context.Background(), input, gollem.WithTemperature(0.3))
	gt.NoError(t, err)

	// Drain channel
	for range resultCh {
	}

	gt.Value(t, len(receivedInput)).Equal(1)
	gt.Value(t, string(receivedInput[0].(gollem.Text))).Equal("stream me")
	gt.Value(t, len(receivedOpts)).Equal(1)

	cfg := gollem.NewGenerateConfig(receivedOpts...)
	gt.Value(t, *cfg.Temperature()).Equal(0.3)
}

func TestGenerateWithFunctionResponseInput(t *testing.T) {
	// Verify that FunctionResponse inputs (previously passed via variadic)
	// work correctly when wrapped in []Input.

	var receivedInput []gollem.Input
	session := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			receivedInput = input
			return &gollem.Response{Texts: []string{"handled"}}, nil
		},
	}

	funcResp := gollem.FunctionResponse{
		ID:   "call_123",
		Name: "my_tool",
		Data: map[string]any{"result": "success"},
	}

	_, err := session.Generate(context.Background(), []gollem.Input{funcResp})
	gt.NoError(t, err)
	gt.Value(t, len(receivedInput)).Equal(1)

	fr, ok := receivedInput[0].(gollem.FunctionResponse)
	gt.True(t, ok)
	gt.Value(t, fr.ID).Equal("call_123")
	gt.Value(t, fr.Name).Equal("my_tool")
	gt.Value(t, fr.Data["result"]).Equal("success")
}

func TestGenerateWithMixedInputTypes(t *testing.T) {
	// Verify that mixed input types (Text + FunctionResponse) in a single
	// []Input slice are correctly passed through.

	var receivedInput []gollem.Input
	session := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			receivedInput = input
			return &gollem.Response{Texts: []string{"done"}}, nil
		},
	}

	input := []gollem.Input{
		gollem.Text("Here are the tool results:"),
		gollem.FunctionResponse{
			ID:   "call_1",
			Name: "tool_a",
			Data: map[string]any{"status": "ok"},
		},
	}

	_, err := session.Generate(context.Background(), input)
	gt.NoError(t, err)
	gt.Value(t, len(receivedInput)).Equal(2)

	_, ok1 := receivedInput[0].(gollem.Text)
	gt.True(t, ok1)
	_, ok2 := receivedInput[1].(gollem.FunctionResponse)
	gt.True(t, ok2)
}

func TestAgentExecuteUsesGenerateWithSliceInput(t *testing.T) {
	// Verify that the Agent (gollem.go) correctly passes []Input
	// when calling session.Generate.

	var receivedInputs [][]gollem.Input
	callCount := 0
	mockClient := &mock.LLMClientMock{
		NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			return &mock.SessionMock{
				GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
					callCount++
					receivedInputs = append(receivedInputs, input)

					if callCount == 1 {
						// First call: return text-only (no tool calls).
						// Default strategy sees no tool calls and returns
						// respond_to_user automatically.
						return &gollem.Response{
							Texts: []string{"hello back"},
						}, nil
					}
					// Fallback for subsequent calls (after respond_to_user handling)
					return &gollem.Response{
						Texts: []string{"done"},
					}, nil
				},
				HistoryFunc: func() (*gollem.History, error) {
					return nil, nil
				},
			}, nil
		},
	}

	agent := gollem.New(mockClient, gollem.WithLoopLimit(5))
	result, err := agent.Execute(context.Background(), gollem.Text("hello agent"))
	gt.NoError(t, err)
	gt.NotNil(t, result)

	// Agent should have called Generate at least once
	gt.True(t, len(receivedInputs) >= 1)
	// The first call should contain the user's input as []Input
	gt.Value(t, len(receivedInputs[0])).Equal(1)
	gt.Value(t, string(receivedInputs[0][0].(gollem.Text))).Equal("hello agent")
}
