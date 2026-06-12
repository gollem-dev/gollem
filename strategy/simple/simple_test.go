package simple_test

import (
	"context"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/strategy/simple"
	"github.com/m-mizutani/gt"
)

func TestSimpleStrategy(t *testing.T) {
	ctx := context.Background()
	strategy := simple.New() // No client needed for simple strategy

	// Initialize strategy
	err := strategy.Init(ctx, []gollem.Input{gollem.Text("test")})
	gt.NoError(t, err)

	t.Run("initial iteration returns InitInput", func(t *testing.T) {
		initInput := []gollem.Input{
			gollem.Text("initial message"),
			gollem.Text("second message"),
		}

		state := &gollem.StrategyState{
			InitInput: initInput,
			Iteration: 0,
		}

		result, _, err := strategy.Handle(ctx, state)
		gt.NoError(t, err)
		gt.Equal(t, len(initInput), len(result))
		gt.Equal(t, initInput[0].String(), result[0].String())
		gt.Equal(t, initInput[1].String(), result[1].String())
	})

	t.Run("subsequent iterations return NextInput", func(t *testing.T) {
		nextInput := []gollem.Input{
			gollem.Text("next message"),
		}

		state := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("initial")},
			NextInput: nextInput,
			Iteration: 1,
		}

		result, _, err := strategy.Handle(ctx, state)
		gt.NoError(t, err)
		gt.Equal(t, len(nextInput), len(result))
		gt.Equal(t, nextInput[0].String(), result[0].String())
	})

	t.Run("returns empty NextInput when no more input", func(t *testing.T) {
		state := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("initial")},
			NextInput: []gollem.Input{},
			Iteration: 2,
		}

		result, _, err := strategy.Handle(ctx, state)
		gt.NoError(t, err)
		gt.Equal(t, 0, len(result))
	})

	t.Run("handles FunctionResponse in NextInput", func(t *testing.T) {
		functionResp := gollem.FunctionResponse{
			ID:   "test-id",
			Name: "test-function",
			Data: map[string]any{"result": "success"},
		}

		state := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("initial")},
			NextInput: []gollem.Input{functionResp},
			Iteration: 3,
		}

		result, _, err := strategy.Handle(ctx, state)
		gt.NoError(t, err)
		gt.Equal(t, 1, len(result))

		// Check if the returned input is the same FunctionResponse
		fr, ok := result[0].(gollem.FunctionResponse)
		gt.True(t, ok)
		gt.Equal(t, functionResp.ID, fr.ID)
		gt.Equal(t, functionResp.Name, fr.Name)
	})
}
