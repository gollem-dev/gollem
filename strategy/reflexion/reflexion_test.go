package reflexion_test

import (
	"context"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/strategy/reflexion"
	"github.com/m-mizutani/gt"
)

// mockLLMClient is a mock implementation of gollem.LLMClient for testing
type mockLLMClient struct {
	sessionFunc func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error)
}

func (m *mockLLMClient) NewSession(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
	if m.sessionFunc != nil {
		return m.sessionFunc(ctx, options...)
	}
	return &mockSession{}, nil
}

func (m *mockLLMClient) GenerateEmbedding(ctx context.Context, dimension int, input []string) ([][]float64, error) {
	return nil, nil
}

// mockSession is a mock implementation of gollem.Session for testing
type mockSession struct {
	generateCount int
}

func (m *mockSession) Generate(ctx context.Context, inputs []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
	m.generateCount++

	// First call: generate a brief response that will fail evaluation
	if m.generateCount == 1 {
		return &gollem.Response{
			Texts: []string{"Brief answer"},
		}, nil
	}

	// Subsequent calls: generate a more detailed response that will pass
	return &gollem.Response{
		Texts: []string{"This is a very detailed and comprehensive answer that should pass the evaluation criteria with sufficient length and detail."},
	}, nil
}

func (m *mockSession) Stream(ctx context.Context, inputs []gollem.Input, opts ...gollem.GenerateOption) (<-chan *gollem.Response, error) {
	return nil, nil
}

func (m *mockSession) GenerateContent(ctx context.Context, input ...gollem.Input) (*gollem.Response, error) {
	return m.Generate(ctx, input)
}

func (m *mockSession) GenerateStream(ctx context.Context, input ...gollem.Input) (<-chan *gollem.Response, error) {
	return m.Stream(ctx, input)
}

func (m *mockSession) History() (*gollem.History, error) {
	return &gollem.History{}, nil
}

func (m *mockSession) AppendHistory(history *gollem.History) error {
	return nil
}

func (m *mockSession) CountToken(ctx context.Context, input ...gollem.Input) (int, error) {
	return 0, nil
}

// TestReflexionStrategy_Success tests that Reflexion succeeds on first try when evaluation passes
func TestReflexionStrategy_Success(t *testing.T) {
	ctx := context.Background()

	// Create a simple evaluator that always succeeds
	evaluator := reflexion.Evaluator(func(ctx context.Context, trajectory *reflexion.Trajectory) (*reflexion.EvaluationResult, error) {
		return &reflexion.EvaluationResult{
			Success: true,
			Score:   1.0,
		}, nil
	})

	client := &mockLLMClient{}
	strategy := reflexion.New(client,
		reflexion.WithMaxTrials(3),
		reflexion.WithEvaluator(evaluator),
	)

	agent := gollem.New(client, gollem.WithStrategy(strategy))

	resp, err := agent.Execute(ctx, gollem.Text("Test question"))
	gt.NoError(t, err)
	gt.True(t, len(resp.Texts) > 0)
}

// TestReflexionStrategy_RetryUntilSuccess tests that Reflexion retries on failure and eventually succeeds
func TestReflexionStrategy_RetryUntilSuccess(t *testing.T) {
	ctx := context.Background()

	attemptCount := 0

	// Evaluator that fails first attempt, succeeds on second
	evaluator := reflexion.Evaluator(func(ctx context.Context, trajectory *reflexion.Trajectory) (*reflexion.EvaluationResult, error) {
		attemptCount++
		if attemptCount == 1 {
			return &reflexion.EvaluationResult{
				Success:  false,
				Feedback: "Response is too brief",
			}, nil
		}
		return &reflexion.EvaluationResult{
			Success: true,
			Score:   1.0,
		}, nil
	})

	client := &mockLLMClient{}
	strategy := reflexion.New(client,
		reflexion.WithMaxTrials(3),
		reflexion.WithMemorySize(2),
		reflexion.WithEvaluator(evaluator),
	)

	agent := gollem.New(client, gollem.WithStrategy(strategy))

	resp, err := agent.Execute(ctx, gollem.Text("Test question"))
	gt.NoError(t, err)
	gt.True(t, len(resp.Texts) > 0)
	gt.Equal(t, attemptCount, 2) // Should have retried once
}

// TestReflexionStrategy_MaxTrialsReached tests that Reflexion stops after max trials
func TestReflexionStrategy_MaxTrialsReached(t *testing.T) {
	ctx := context.Background()

	attemptCount := 0

	// Evaluator that always fails
	evaluator := reflexion.Evaluator(func(ctx context.Context, trajectory *reflexion.Trajectory) (*reflexion.EvaluationResult, error) {
		attemptCount++
		return &reflexion.EvaluationResult{
			Success:  false,
			Feedback: "Always fails",
		}, nil
	})

	client := &mockLLMClient{}
	strategy := reflexion.New(client,
		reflexion.WithMaxTrials(2),
		reflexion.WithEvaluator(evaluator),
	)

	agent := gollem.New(client, gollem.WithStrategy(strategy))

	resp, err := agent.Execute(ctx, gollem.Text("Test question"))
	gt.NoError(t, err)
	gt.True(t, len(resp.Texts) > 0) // Should still return final response
	gt.Equal(t, attemptCount, 2)    // Should have tried exactly max trials
}

// TestReflexionStrategy_WithHooks tests that hooks are called at appropriate times
func TestReflexionStrategy_WithHooks(t *testing.T) {
	ctx := context.Background()

	var (
		trialStartCalls    []int
		trialEndCalls      []int
		reflectionGenCalls []int
	)

	hooks := &testHooks{
		onTrialStart: func(ctx context.Context, trialNum int) error {
			trialStartCalls = append(trialStartCalls, trialNum)
			return nil
		},
		onTrialEnd: func(ctx context.Context, trialNum int, evaluation *reflexion.EvaluationResult) error {
			trialEndCalls = append(trialEndCalls, trialNum)
			return nil
		},
		onReflectionGenerated: func(ctx context.Context, trialNum int, reflection string) error {
			reflectionGenCalls = append(reflectionGenCalls, trialNum)
			return nil
		},
	}

	attemptCount := 0
	evaluator := reflexion.Evaluator(func(ctx context.Context, trajectory *reflexion.Trajectory) (*reflexion.EvaluationResult, error) {
		attemptCount++
		if attemptCount == 1 {
			return &reflexion.EvaluationResult{Success: false, Feedback: "Retry"}, nil
		}
		return &reflexion.EvaluationResult{Success: true}, nil
	})

	client := &mockLLMClient{}
	strategy := reflexion.New(client,
		reflexion.WithMaxTrials(3),
		reflexion.WithEvaluator(evaluator),
		reflexion.WithHooks(hooks),
	)

	agent := gollem.New(client, gollem.WithStrategy(strategy))

	_, err := agent.Execute(ctx, gollem.Text("Test question"))
	gt.NoError(t, err)

	// Verify hooks were called
	gt.Equal(t, len(trialStartCalls), 2)    // Trial 1 and 2
	gt.Equal(t, len(trialEndCalls), 2)      // End of trial 1 and 2
	gt.Equal(t, len(reflectionGenCalls), 1) // Reflection after trial 1 failure
}

// testHooks is a helper struct for testing hooks
type testHooks struct {
	onTrialStart          func(ctx context.Context, trialNum int) error
	onTrialEnd            func(ctx context.Context, trialNum int, evaluation *reflexion.EvaluationResult) error
	onReflectionGenerated func(ctx context.Context, trialNum int, reflection string) error
}

func (h *testHooks) OnTrialStart(ctx context.Context, trialNum int) error {
	if h.onTrialStart != nil {
		return h.onTrialStart(ctx, trialNum)
	}
	return nil
}

func (h *testHooks) OnTrialEnd(ctx context.Context, trialNum int, evaluation *reflexion.EvaluationResult) error {
	if h.onTrialEnd != nil {
		return h.onTrialEnd(ctx, trialNum, evaluation)
	}
	return nil
}

func (h *testHooks) OnReflectionGenerated(ctx context.Context, trialNum int, reflection string) error {
	if h.onReflectionGenerated != nil {
		return h.onReflectionGenerated(ctx, trialNum, reflection)
	}
	return nil
}
