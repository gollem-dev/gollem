package gollem_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/mock"
	"github.com/m-mizutani/goerr/v2"
	"github.com/m-mizutani/gt"
)

type testQueryResult struct {
	Name  string `json:"name" description:"name of the item"`
	Count int    `json:"count" description:"number of items"`
}

type testQueryResultWithConstraints struct {
	Status string `json:"status" description:"status of the item" enum:"active,inactive" required:"true"`
	Score  int    `json:"score" description:"score value" min:"0" max:"100"`
}

func setupQueryMock(t *testing.T, genFunc func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error)) *mock.LLMClientMock {
	t.Helper()
	sessionMock := &mock.SessionMock{
		GenerateFunc: genFunc,
	}
	return &mock.LLMClientMock{
		NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			return sessionMock, nil
		},
	}
}

func TestQuerySuccess(t *testing.T) {
	client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
		return &gollem.Response{
			Texts:       []string{`{"name":"test","count":42}`},
			InputToken:  10,
			OutputToken: 5,
		}, nil
	})

	resp, err := gollem.Query[testQueryResult](context.Background(), client, "test prompt")
	gt.NoError(t, err)
	gt.Value(t, resp.Data.Name).Equal("test")
	gt.Value(t, resp.Data.Count).Equal(42)
	gt.Value(t, resp.InputToken).Equal(10)
	gt.Value(t, resp.OutputToken).Equal(5)
}

func buildSessionConfig(opts []gollem.SessionOption) gollem.SessionConfig {
	return gollem.NewSessionConfig(opts...)
}

func TestQueryWithSystemPrompt(t *testing.T) {
	var capturedOpts []gollem.SessionOption
	sessionMock := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			return &gollem.Response{
				Texts: []string{`{"name":"x","count":1}`},
			}, nil
		},
	}
	client := &mock.LLMClientMock{
		NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			capturedOpts = options
			return sessionMock, nil
		},
	}

	_, err := gollem.Query[testQueryResult](context.Background(), client, "test",
		gollem.WithQuerySystemPrompt("You are a planner"),
	)
	gt.NoError(t, err)

	cfg := buildSessionConfig(capturedOpts)
	gt.Value(t, cfg.ContentType()).Equal(gollem.ContentTypeJSON)
	gt.Value(t, cfg.ResponseSchema()).NotEqual((*gollem.Parameter)(nil))
	gt.Value(t, cfg.SystemPrompt()).Equal("You are a planner")
	gt.Value(t, cfg.History()).Equal((*gollem.History)(nil))
}

func TestQueryWithHistory(t *testing.T) {
	var capturedOpts []gollem.SessionOption
	sessionMock := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			return &gollem.Response{
				Texts: []string{`{"name":"x","count":1}`},
			}, nil
		},
	}
	client := &mock.LLMClientMock{
		NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			capturedOpts = options
			return sessionMock, nil
		},
	}

	history := &gollem.History{}
	_, err := gollem.Query[testQueryResult](context.Background(), client, "test",
		gollem.WithQueryHistory(history),
	)
	gt.NoError(t, err)

	cfg := buildSessionConfig(capturedOpts)
	gt.Value(t, cfg.ContentType()).Equal(gollem.ContentTypeJSON)
	gt.Value(t, cfg.ResponseSchema()).NotEqual((*gollem.Parameter)(nil))
	gt.Value(t, cfg.SystemPrompt()).Equal("")
	gt.Value(t, cfg.History()).Equal(history)
}

func TestQueryRetrySuccess(t *testing.T) {
	callCount := 0
	client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
		callCount++
		if callCount == 1 {
			// First call returns invalid JSON
			return &gollem.Response{
				Texts:       []string{`not valid json`},
				InputToken:  10,
				OutputToken: 5,
			}, nil
		}
		// Second call returns valid JSON
		return &gollem.Response{
			Texts:       []string{`{"name":"retry","count":99}`},
			InputToken:  12,
			OutputToken: 6,
		}, nil
	})

	resp, err := gollem.Query[testQueryResult](context.Background(), client, "test")
	gt.NoError(t, err)
	gt.Value(t, resp.Data.Name).Equal("retry")
	gt.Value(t, resp.Data.Count).Equal(99)
	gt.Value(t, callCount).Equal(2)
	// Token counts should be accumulated
	gt.Value(t, resp.InputToken).Equal(22)
	gt.Value(t, resp.OutputToken).Equal(11)
}

func TestQueryRetryExhausted(t *testing.T) {
	callCount := 0
	client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
		callCount++
		return &gollem.Response{
			Texts: []string{`not json`},
		}, nil
	})

	_, err := gollem.Query[testQueryResult](context.Background(), client, "test",
		gollem.WithQueryMaxRetry(2),
	)
	gt.Error(t, err)
	// 1 initial + 2 retries = 3 calls
	gt.Value(t, callCount).Equal(3)
}

// encoding/json decodes "Status" into the field named "status", so a response
// using that key must fail validation rather than deliver an unvalidated value.
func TestQueryRejectsCaseVariantKey(t *testing.T) {
	var retryPrompt string
	callCount := 0
	client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
		callCount++
		if callCount == 1 {
			return &gollem.Response{Texts: []string{`{"status":"active","Status":"deleted"}`}}, nil
		}
		if text, ok := input[0].(gollem.Text); ok {
			retryPrompt = string(text)
		}
		return &gollem.Response{Texts: []string{`{"status":"inactive"}`}}, nil
	})

	resp, err := gollem.Query[testQueryResultWithConstraints](context.Background(), client, "test")
	gt.NoError(t, err)
	gt.Value(t, callCount).Equal(2)
	gt.Value(t, resp.Data.Status).Equal("inactive")
	gt.S(t, retryPrompt).Contains(`key "root.Status" differs from declared parameter "status"`)
}

func TestQueryEmptyResponse(t *testing.T) {
	client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
		return &gollem.Response{
			Texts: []string{},
		}, nil
	})

	_, err := gollem.Query[testQueryResult](context.Background(), client, "test")
	gt.Error(t, err)
}

func TestIsNaturalFinishReason(t *testing.T) {
	// The natural finish reason of each provider, as documented on
	// Response.FinishReason.
	for _, reason := range []string{"end_turn", "STOP", "stop", "completed"} {
		gt.True(t, gollem.IsNaturalFinishReason(reason))
	}
	for _, reason := range []string{"", "max_tokens", "tool_use", "refusal", "MAX_TOKENS", "SAFETY", "length", "tool_calls", "max_output_tokens"} {
		gt.False(t, gollem.IsNaturalFinishReason(reason))
	}
}

// thinkingOnlyResponse is a response that ended the turn with only a thinking
// block whose text was omitted.
func thinkingOnlyResponse() *gollem.Response {
	return &gollem.Response{
		FinishReason:    "end_turn",
		OtherBlockTypes: []string{"thinking"},
		InputToken:      10,
		OutputToken:     1300,
	}
}

func TestQueryContinuesAfterResponseWithoutText(t *testing.T) {
	var inputs [][]gollem.Input
	client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
		inputs = append(inputs, input)
		if len(inputs) == 1 {
			return thinkingOnlyResponse(), nil
		}
		return &gollem.Response{
			Texts:        []string{`{"name":"ok","count":1}`},
			FinishReason: "end_turn",
			InputToken:   12,
			OutputToken:  6,
		}, nil
	})

	resp, err := gollem.Query[testQueryResult](context.Background(), client, "test")
	gt.NoError(t, err).Required()
	gt.Equal(t, "ok", resp.Data.Name)
	gt.Equal(t, 22, resp.InputToken)
	gt.Equal(t, 1306, resp.OutputToken)

	gt.A(t, inputs).Length(2).Required()
	gt.Equal(t, []gollem.Input{gollem.Text(gollem.ContinuePrompt)}, inputs[1])
}

func TestQueryResponseWithoutTextExhausted(t *testing.T) {
	callCount := 0
	client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
		callCount++
		return thinkingOnlyResponse(), nil
	})

	_, err := gollem.Query[testQueryResult](context.Background(), client, "test",
		gollem.WithQueryMaxRetry(2),
	)
	gt.Error(t, err).Required()
	// 1 initial + 2 retries = 3 calls
	gt.Equal(t, 3, callCount)

	values := goerr.Values(err)
	gt.Equal(t, any(3), values["attempt"])
	gt.Equal(t, any("end_turn"), values["finish_reason"])
	gt.Equal(t, any([]string{"thinking"}), values["other_block_types"])
	gt.Equal(t, any(1300), values["output_tokens"])
}

func TestQueryResponseWithoutTextNotContinued(t *testing.T) {
	type testCase struct {
		resp *gollem.Response
	}
	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			callCount := 0
			client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
				callCount++
				return tc.resp, nil
			})

			_, err := gollem.Query[testQueryResult](context.Background(), client, "test")
			gt.Error(t, err).Required()
			gt.Equal(t, 1, callCount)
			gt.Equal(t, any(tc.resp.FinishReason), goerr.Values(err)["finish_reason"])
		}
	}

	t.Run("token limit", runTest(testCase{resp: &gollem.Response{
		FinishReason:    "max_tokens",
		OtherBlockTypes: []string{"thinking"},
	}}))
	t.Run("no finish reason", runTest(testCase{resp: &gollem.Response{}}))
	t.Run("function call with a natural finish reason", runTest(testCase{resp: &gollem.Response{
		FinishReason:  "STOP",
		FunctionCalls: []*gollem.FunctionCall{{ID: "call_1", Name: "lookup"}},
	}}))
}

func TestQueryRefusal(t *testing.T) {
	type testCase struct {
		texts []string
	}
	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			callCount := 0
			client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
				callCount++
				return &gollem.Response{
					Texts:        tc.texts,
					FinishReason: "refusal",
					Refusal: &gollem.Refusal{
						Reason:      "refusal",
						Categories:  []string{"general_harms"},
						Explanation: "The request was declined.",
					},
				}, nil
			})

			_, err := gollem.Query[testQueryResult](context.Background(), client, "test")
			gt.Error(t, err).Required()
			gt.True(t, errors.Is(err, gollem.ErrProhibitedContent))
			// A refusal is not retried with a JSON correction prompt.
			gt.Equal(t, 1, callCount)

			values := goerr.Values(err)
			gt.Equal(t, any("refusal"), values["finish_reason"])
			gt.Equal(t, any("refusal"), values["refusal_reason"])
			gt.Equal(t, any([]string{"general_harms"}), values["refusal_categories"])
			gt.Equal(t, any("The request was declined."), values["refusal_explanation"])
		}
	}

	t.Run("without text", runTest(testCase{texts: []string{}}))
	t.Run("with a partial text", runTest(testCase{texts: []string{`{"name":`}}))
}

func TestQueryGenerateError(t *testing.T) {
	callCount := 0
	sessionMock := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			callCount++
			return nil, errors.New("network error")
		},
	}
	clientWithCounter := &mock.LLMClientMock{
		NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			return sessionMock, nil
		},
	}

	_, err := gollem.Query[testQueryResult](context.Background(), clientWithCounter, "test")
	gt.Error(t, err)
	// Should not retry on Generate error
	gt.Value(t, callCount).Equal(1)
}

func TestQueryNewSessionError(t *testing.T) {
	client := &mock.LLMClientMock{
		NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			return nil, errors.New("session creation failed")
		},
	}

	_, err := gollem.Query[testQueryResult](context.Background(), client, "test")
	gt.Error(t, err)
}

func TestQueryNilSession(t *testing.T) {
	client := &mock.LLMClientMock{
		NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			return nil, nil
		},
	}

	_, err := gollem.Query[testQueryResult](context.Background(), client, "test")
	gt.Error(t, err)
}

func TestQueryNegativeMaxRetry(t *testing.T) {
	callCount := 0
	client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
		callCount++
		return &gollem.Response{
			Texts: []string{`{"name":"ok","count":1}`},
		}, nil
	})

	// Negative retry should still make the initial request
	resp, err := gollem.Query[testQueryResult](context.Background(), client, "test",
		gollem.WithQueryMaxRetry(-5),
	)
	gt.NoError(t, err)
	gt.Value(t, resp.Data.Name).Equal("ok")
	gt.Value(t, callCount).Equal(1)
}

func TestQueryRetryFeedbackMessage(t *testing.T) {
	var secondCallInputs []gollem.Input
	callCount := 0
	client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
		callCount++
		if callCount == 1 {
			return &gollem.Response{
				Texts: []string{`{broken`},
			}, nil
		}
		secondCallInputs = input
		return &gollem.Response{
			Texts: []string{`{"name":"ok","count":1}`},
		}, nil
	})

	_, err := gollem.Query[testQueryResult](context.Background(), client, "test")
	gt.NoError(t, err)
	// Verify that the retry input contains error feedback
	gt.Value(t, len(secondCallInputs)).Equal(1)
	feedbackText := secondCallInputs[0].String()
	gt.Value(t, feedbackText != "").Equal(true)
}

func TestQuerySchemaValidationEnumViolation(t *testing.T) {
	callCount := 0
	client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
		callCount++
		if callCount == 1 {
			// Valid JSON but enum constraint violation
			return &gollem.Response{
				Texts:       []string{`{"status":"unknown","score":50}`},
				InputToken:  10,
				OutputToken: 5,
			}, nil
		}
		// Second call returns valid data
		return &gollem.Response{
			Texts:       []string{`{"status":"active","score":50}`},
			InputToken:  12,
			OutputToken: 6,
		}, nil
	})

	resp, err := gollem.Query[testQueryResultWithConstraints](context.Background(), client, "test")
	gt.NoError(t, err)
	gt.Value(t, resp.Data.Status).Equal("active")
	gt.Value(t, resp.Data.Score).Equal(50)
	gt.Value(t, callCount).Equal(2)
}

func TestQuerySchemaValidationRequiredMissing(t *testing.T) {
	callCount := 0
	client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
		callCount++
		if callCount == 1 {
			// Missing required "status" field
			return &gollem.Response{
				Texts:       []string{`{"score":50}`},
				InputToken:  10,
				OutputToken: 5,
			}, nil
		}
		return &gollem.Response{
			Texts:       []string{`{"status":"active","score":50}`},
			InputToken:  12,
			OutputToken: 6,
		}, nil
	})

	resp, err := gollem.Query[testQueryResultWithConstraints](context.Background(), client, "test")
	gt.NoError(t, err)
	gt.Value(t, resp.Data.Status).Equal("active")
	gt.Value(t, callCount).Equal(2)
}

func TestQuerySchemaValidationExhausted(t *testing.T) {
	callCount := 0
	client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
		callCount++
		// Always return invalid enum value
		return &gollem.Response{
			Texts: []string{`{"status":"unknown","score":50}`},
		}, nil
	})

	_, err := gollem.Query[testQueryResultWithConstraints](context.Background(), client, "test",
		gollem.WithQueryMaxRetry(2),
	)
	gt.Error(t, err)
	// 1 initial + 2 retries = 3 calls
	gt.Value(t, callCount).Equal(3)
}

func TestQuerySchemaValidationSuccess(t *testing.T) {
	client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
		return &gollem.Response{
			Texts:       []string{`{"status":"active","score":80}`},
			InputToken:  10,
			OutputToken: 5,
		}, nil
	})

	resp, err := gollem.Query[testQueryResultWithConstraints](context.Background(), client, "test")
	gt.NoError(t, err)
	gt.Value(t, resp.Data.Status).Equal("active")
	gt.Value(t, resp.Data.Score).Equal(80)
}

type testQueryResultWithMap struct {
	Query  string            `json:"query" required:"true"`
	Labels map[string]string `json:"labels"`
}

func TestQueryWithMap(t *testing.T) {
	t.Run("schema keeps the map and the response decodes into it", func(t *testing.T) {
		var capturedOpts []gollem.SessionOption
		sessionMock := &mock.SessionMock{
			GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
				return &gollem.Response{Texts: []string{`{"query":"q","labels":{"a":"x","b":"y"}}`}}, nil
			},
		}
		client := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				capturedOpts = options
				return sessionMock, nil
			},
		}

		resp, err := gollem.Query[testQueryResultWithMap](context.Background(), client, "test")
		gt.NoError(t, err)
		gt.Equal(t, map[string]string{"a": "x", "b": "y"}, resp.Data.Labels)

		cfg := buildSessionConfig(capturedOpts)
		labels := cfg.ResponseSchema().Properties["labels"]
		gt.Equal(t, gollem.TypeObject, labels.Type)
		gt.Nil(t, labels.Properties)
		gt.Equal(t, gollem.TypeString, labels.AdditionalProperties.Type)
	})

	t.Run("map value that breaks the value schema is fed back for a retry", func(t *testing.T) {
		type item struct {
			Name string `json:"name" required:"true"`
		}
		type result struct {
			Items map[string]item `json:"items"`
		}

		var retryInput string
		callCount := 0
		client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			callCount++
			if callCount == 1 {
				// Decodes into result, but the value of "a" lacks the required name.
				return &gollem.Response{Texts: []string{`{"items":{"a":{}}}`}}, nil
			}
			retryInput = string(input[0].(gollem.Text))
			return &gollem.Response{Texts: []string{`{"items":{"a":{"name":"x"}}}`}}, nil
		})

		resp, err := gollem.Query[result](context.Background(), client, "test")
		gt.NoError(t, err)
		gt.Equal(t, 2, callCount)
		gt.Equal(t, "x", resp.Data.Items["a"].Name)
		gt.S(t, retryInput).Contains("did not match the schema constraints")
		gt.S(t, retryInput).Contains("required parameter missing")
	})

	t.Run("null map value is fed back for a retry", func(t *testing.T) {
		var retryInput string
		callCount := 0
		client := setupQueryMock(t, func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			callCount++
			if callCount == 1 {
				// Decodes into "" for map[string]string, so only validation catches it.
				return &gollem.Response{Texts: []string{`{"query":"q","labels":{"a":null}}`}}, nil
			}
			retryInput = string(input[0].(gollem.Text))
			return &gollem.Response{Texts: []string{`{"query":"q","labels":{"a":"x"}}`}}, nil
		})

		resp, err := gollem.Query[testQueryResultWithMap](context.Background(), client, "test")
		gt.NoError(t, err)
		gt.Equal(t, 2, callCount)
		gt.Equal(t, map[string]string{"a": "x"}, resp.Data.Labels)
		gt.S(t, retryInput).Contains("map value must not be null")
	})

	t.Run("unsupported schema error from the provider reaches the caller", func(t *testing.T) {
		client := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return nil, fmt.Errorf("convert response schema: %w", gollem.ErrUnsupportedSchema)
			},
		}

		_, err := gollem.Query[testQueryResultWithMap](context.Background(), client, "test")
		gt.True(t, errors.Is(err, gollem.ErrUnsupportedSchema))
	})
}
