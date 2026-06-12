package simple

import (
	"context"

	"github.com/gollem-dev/gollem"
)

// simpleStrategy implements a simple loop strategy that continues as long as there is input
// Note: This implementation should be kept in sync with defaultStrategy in ../../strategy.go
type simpleStrategy struct{}

// New creates a simple loop strategy that continues as long as there is input
func New() *simpleStrategy {
	return &simpleStrategy{}
}

func (s *simpleStrategy) Init(ctx context.Context, inputs []gollem.Input) error {
	// Simple strategy needs no special initialization
	return nil
}

func (s *simpleStrategy) Handle(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
	if state.Iteration == 0 {
		return state.InitInput, nil, nil
	}

	// Check LLM's last response for termination condition
	if state.LastResponse != nil {
		if len(state.LastResponse.FunctionCalls) == 0 {
			// No tool calls = final response, use as conclusion
			executeResponse := &gollem.ExecuteResponse{
				Texts: state.LastResponse.Texts,
			}
			return nil, executeResponse, nil
		}
	}

	return state.NextInput, nil, nil
}

func (s *simpleStrategy) Tools(ctx context.Context) ([]gollem.Tool, error) {
	// Simple strategy provides no additional tools
	return []gollem.Tool{}, nil
}
