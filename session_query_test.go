package gollem_test

import (
	"context"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/mock"
	"github.com/m-mizutani/gt"
)

type testSessionQueryResult struct {
	Answer string `json:"answer"`
}

func TestSessionQuerySuccess(t *testing.T) {
	session := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			return &gollem.Response{
				Texts:       []string{`{"answer":"hello"}`},
				InputToken:  10,
				OutputToken: 5,
			}, nil
		},
	}

	resp, err := gollem.SessionQuery[testSessionQueryResult](context.Background(), session, "test prompt")
	gt.NoError(t, err)
	gt.Value(t, resp.Data.Answer).Equal("hello")
	gt.Value(t, resp.InputToken).Equal(10)
	gt.Value(t, resp.OutputToken).Equal(5)
}

func TestSessionQueryRetry(t *testing.T) {
	callCount := 0
	session := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			callCount++
			if callCount == 1 {
				return &gollem.Response{
					Texts:       []string{`not json`},
					InputToken:  10,
					OutputToken: 5,
				}, nil
			}
			return &gollem.Response{
				Texts:       []string{`{"answer":"retried"}`},
				InputToken:  10,
				OutputToken: 5,
			}, nil
		},
	}

	resp, err := gollem.SessionQuery[testSessionQueryResult](context.Background(), session, "test prompt")
	gt.NoError(t, err)
	gt.Value(t, resp.Data.Answer).Equal("retried")
	gt.Value(t, callCount).Equal(2)
}

func TestSessionQueryNilSession(t *testing.T) {
	_, err := gollem.SessionQuery[testSessionQueryResult](context.Background(), nil, "test prompt")
	gt.Error(t, err)
}

func TestSessionQueryPassesResponseSchemaAsGenerateOption(t *testing.T) {
	var receivedOpts []gollem.GenerateOption
	session := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			receivedOpts = opts
			return &gollem.Response{
				Texts: []string{`{"answer":"ok"}`},
			}, nil
		},
	}

	_, err := gollem.SessionQuery[testSessionQueryResult](context.Background(), session, "test prompt")
	gt.NoError(t, err)

	// SessionQuery should pass exactly one GenerateOption: WithGenerateResponseSchema
	gt.Value(t, len(receivedOpts)).Equal(1)

	// Build config from the options and verify the schema is populated
	cfg := gollem.NewGenerateConfig(receivedOpts...)
	gt.NotNil(t, cfg.ResponseSchema())
	// The schema should reflect testSessionQueryResult's structure
	gt.Value(t, cfg.ResponseSchema().Type).Equal(gollem.TypeObject)
	props := cfg.ResponseSchema().Properties
	gt.NotNil(t, props["answer"])
	gt.Value(t, props["answer"].Type).Equal(gollem.TypeString)
}

func TestSessionQueryPerCallOptionsOverrideSessionDefaults(t *testing.T) {
	// Simulate a session created with ContentTypeText (the default).
	// SessionQuery must pass WithGenerateResponseSchema so the LLM
	// returns JSON regardless of the session's default ContentType.
	//
	// We verify this by inspecting the GenerateOptions the mock receives.

	var allCallOpts [][]gollem.GenerateOption
	session := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			allCallOpts = append(allCallOpts, opts)
			return &gollem.Response{
				Texts: []string{`{"answer":"structured"}`},
			}, nil
		},
	}

	resp, err := gollem.SessionQuery[testSessionQueryResult](context.Background(), session, "give me structured data")
	gt.NoError(t, err)
	gt.Value(t, resp.Data.Answer).Equal("structured")

	// Exactly one call should have been made
	gt.Value(t, len(allCallOpts)).Equal(1)

	// The per-call options must include a response schema
	cfg := gollem.NewGenerateConfig(allCallOpts[0]...)
	gt.NotNil(t, cfg.ResponseSchema())
}

func TestSessionQueryPreservesConversationHistory(t *testing.T) {
	// The key value of SessionQuery over Query is that it reuses an
	// existing session, so conversation history is preserved.
	// We simulate this by tracking all Generate calls and their inputs.

	var allInputs [][]gollem.Input
	session := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			allInputs = append(allInputs, input)
			return &gollem.Response{
				Texts: []string{`{"answer":"ok"}`},
			}, nil
		},
	}

	// First call: regular Generate (simulating prior conversation)
	_, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("My name is Alice")})
	gt.NoError(t, err)

	// Second call: SessionQuery on the same session
	resp, err := gollem.SessionQuery[testSessionQueryResult](context.Background(), session, "What is my name?")
	gt.NoError(t, err)
	gt.Value(t, resp.Data.Answer).Equal("ok")

	// Both calls should have gone to the same session's Generate method
	gt.Value(t, len(allInputs)).Equal(2)

	// First call input
	gt.Value(t, len(allInputs[0])).Equal(1)
	gt.Value(t, string(allInputs[0][0].(gollem.Text))).Equal("My name is Alice")

	// Second call input (from SessionQuery)
	gt.Value(t, len(allInputs[1])).Equal(1)
	gt.Value(t, string(allInputs[1][0].(gollem.Text))).Equal("What is my name?")
}

func TestSessionQueryDoesNotCreateNewSession(t *testing.T) {
	// SessionQuery must NOT call NewSession. It operates on the
	// existing session. We verify by using a SessionMock directly
	// (not an LLMClientMock) and confirming Generate is called.

	generateCalled := false
	session := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			generateCalled = true
			return &gollem.Response{
				Texts: []string{`{"answer":"direct"}`},
			}, nil
		},
	}

	resp, err := gollem.SessionQuery[testSessionQueryResult](context.Background(), session, "test")
	gt.NoError(t, err)
	gt.True(t, generateCalled)
	gt.Value(t, resp.Data.Answer).Equal("direct")

	// Verify Generate was called exactly once
	calls := session.GenerateCalls()
	gt.Value(t, len(calls)).Equal(1)
}

func TestSessionQueryRetryPreservesPerCallOptions(t *testing.T) {
	// On retry, the per-call GenerateOptions (especially ResponseSchema)
	// must still be passed. Otherwise the LLM might not return JSON.

	callCount := 0
	var retryOpts []gollem.GenerateOption
	session := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			callCount++
			if callCount == 1 {
				return &gollem.Response{
					Texts: []string{`broken`},
				}, nil
			}
			retryOpts = opts
			return &gollem.Response{
				Texts: []string{`{"answer":"fixed"}`},
			}, nil
		},
	}

	resp, err := gollem.SessionQuery[testSessionQueryResult](context.Background(), session, "test")
	gt.NoError(t, err)
	gt.Value(t, resp.Data.Answer).Equal("fixed")
	gt.Value(t, callCount).Equal(2)

	// The retry call must still carry the response schema option
	cfg := gollem.NewGenerateConfig(retryOpts...)
	gt.NotNil(t, cfg.ResponseSchema())
}

func TestSessionQueryMaxRetryOption(t *testing.T) {
	callCount := 0
	session := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			callCount++
			return &gollem.Response{
				Texts: []string{`not json`},
			}, nil
		},
	}

	_, err := gollem.SessionQuery[testSessionQueryResult](
		context.Background(), session, "test",
		gollem.WithSessionQueryMaxRetry(1),
	)
	gt.Error(t, err)
	// maxRetry=1 means: 1 initial attempt + 1 retry = 2 calls total
	gt.Value(t, callCount).Equal(2)
}

func TestSessionQueryTokenAccumulation(t *testing.T) {
	callCount := 0
	session := &mock.SessionMock{
		GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			callCount++
			if callCount == 1 {
				return &gollem.Response{
					Texts:       []string{`invalid`},
					InputToken:  100,
					OutputToken: 50,
				}, nil
			}
			return &gollem.Response{
				Texts:       []string{`{"answer":"ok"}`},
				InputToken:  80,
				OutputToken: 30,
			}, nil
		},
	}

	resp, err := gollem.SessionQuery[testSessionQueryResult](context.Background(), session, "test")
	gt.NoError(t, err)
	// Tokens should accumulate across retries
	gt.Value(t, resp.InputToken).Equal(180)
	gt.Value(t, resp.OutputToken).Equal(80)
}
