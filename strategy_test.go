package gollem_test

import (
	"context"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/mock"
	"github.com/m-mizutani/gt"
)

// testStrategy implements Strategy interface for testing
type testStrategy struct {
	client  gollem.LLMClient
	handler func(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error)
}

func (s *testStrategy) Init(ctx context.Context, inputs []gollem.Input) error {
	return nil
}

func (s *testStrategy) Handle(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
	return s.handler(ctx, state)
}

func (s *testStrategy) Tools(ctx context.Context) ([]gollem.Tool, error) {
	return []gollem.Tool{}, nil
}

func TestStrategyInterface(t *testing.T) {
	t.Run("strategy returns proper signature", func(t *testing.T) {
		strategy := &testStrategy{
			client: &mock.LLMClientMock{},
			handler: func(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
				// First iteration returns input
				if state.Iteration == 0 {
					return state.InitInput, nil, nil
				}
				// Second iteration returns conclusion
				return nil, gollem.NewExecuteResponse("Strategy completed"), nil
			},
		}

		// Test first call
		state := &gollem.StrategyState{
			Iteration: 0,
			InitInput: []gollem.Input{gollem.Text("test")},
		}
		inputs, response, err := strategy.Handle(context.Background(), state)
		gt.NoError(t, err)
		gt.Equal(t, []gollem.Input{gollem.Text("test")}, inputs)
		gt.Nil(t, response)

		// Test second call
		state.Iteration = 1
		inputs, response, err = strategy.Handle(context.Background(), state)
		gt.NoError(t, err)
		gt.Nil(t, inputs)
		gt.NotNil(t, response)
		gt.Equal(t, "Strategy completed", response.String())
	})

	t.Run("strategy error handling", func(t *testing.T) {
		strategy := &testStrategy{
			client: &mock.LLMClientMock{},
			handler: func(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
				if state.Iteration == 0 {
					return nil, nil, gollem.ErrLoopLimitExceeded
				}
				return nil, gollem.NewExecuteResponse("Should not reach"), nil
			},
		}

		state := &gollem.StrategyState{
			Iteration: 0,
		}
		inputs, response, err := strategy.Handle(context.Background(), state)
		gt.Error(t, err)
		gt.Nil(t, inputs)
		gt.Nil(t, response)
		gt.Equal(t, gollem.ErrLoopLimitExceeded, err)
	})
}

func TestCustomStrategies(t *testing.T) {
	t.Run("immediate conclusion strategy", func(t *testing.T) {
		strategy := &testStrategy{
			client: &mock.LLMClientMock{},
			handler: func(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
				// Always return immediate conclusion
				return nil, gollem.NewExecuteResponse("Immediate conclusion"), nil
			},
		}

		llmClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{}, nil
			},
		}
		agent := gollem.New(llmClient, gollem.WithStrategy(strategy))
		result, err := agent.Execute(context.Background(), gollem.Text("test"))

		gt.NoError(t, err)
		gt.NotNil(t, result)
		gt.Equal(t, "Immediate conclusion", result.String())
	})

	t.Run("conditional conclusion strategy", func(t *testing.T) {
		callCount := 0
		strategy := &testStrategy{
			client: &mock.LLMClientMock{},
			handler: func(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
				callCount++
				if callCount >= 3 {
					return nil, gollem.NewExecuteResponse("Completed after 3 iterations"), nil
				}
				return []gollem.Input{gollem.Text("continue iteration")}, nil, nil
			},
		}

		// Mock LLM client
		mockClient := &mock.LLMClientMock{}
		mockSession := &mock.SessionMock{}
		mockSession.GenerateFunc = func(ctx context.Context, inputs []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			return &gollem.Response{
				Texts:         []string{"LLM response"},
				FunctionCalls: []*gollem.FunctionCall{},
			}, nil
		}
		mockClient.NewSessionFunc = func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			return mockSession, nil
		}

		agent := gollem.New(mockClient, gollem.WithStrategy(strategy))
		result, err := agent.Execute(context.Background(), gollem.Text("test"))

		gt.NoError(t, err)
		gt.NotNil(t, result)
		gt.Equal(t, "Completed after 3 iterations", result.String())
		gt.Equal(t, 3, callCount)
	})

	t.Run("multiple texts in ExecuteResponse", func(t *testing.T) {
		strategy := &testStrategy{
			client: &mock.LLMClientMock{},
			handler: func(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
				return nil, gollem.NewExecuteResponse("First part", "Second part", "Third part"), nil
			},
		}

		llmClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{}, nil
			},
		}
		agent := gollem.New(llmClient, gollem.WithStrategy(strategy))
		result, err := agent.Execute(context.Background(), gollem.Text("test"))

		gt.NoError(t, err)
		gt.NotNil(t, result)
		gt.Equal(t, "First part Second part Third part", result.String())
		gt.Equal(t, []string{"First part", "Second part", "Third part"}, result.Texts)
	})
}
