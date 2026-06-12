// Package reflexion implements the Reflexion strategy for gollem.
//
// Reflexion is a framework for language agents that learn through verbal feedback
// and self-reflection. It enables agents to improve their performance across
// multiple trials by maintaining episodic memory of past reflections.
//
// Basic usage:
//
//	strategy := reflexion.New(llmClient,
//	    reflexion.WithMaxTrials(3),
//	    reflexion.WithMemorySize(3),
//	)
//
//	agent := gollem.New(llmClient, gollem.WithStrategy(strategy))
//	response, err := agent.Execute(ctx, gollem.Text("Solve this task..."))
//
// The strategy will:
//  1. Execute a trial with the given input
//  2. Evaluate the result using the configured evaluator
//  3. If failed, generate a reflection and try again with the reflection as context
//  4. Repeat until success or max trials reached
package reflexion

import (
	"context"

	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/goerr/v2"
)

const (
	// DefaultMaxTrials is the default maximum number of trials
	DefaultMaxTrials = 3
	// DefaultMemorySize is the default maximum size of episodic memory
	DefaultMemorySize = 3
)

// Strategy is the main Reflexion strategy implementation.
// It manages multiple trials, evaluates their results, generates reflections,
// and maintains episodic memory to improve performance across trials.
type Strategy struct {
	client    gollem.LLMClient
	evaluator Evaluator
	hooks     Hooks

	// Configuration
	maxTrials  int
	memorySize int

	// Internal state (reset on Init)
	currentTrial int
	trials       []*trial
	memory       *memory
}

// New creates a new Reflexion strategy with the given LLM client and options.
// By default, it uses LLMEvaluator for evaluation, max 3 trials, and memory size of 3.
func New(client gollem.LLMClient, options ...Option) *Strategy {
	s := &Strategy{
		client:     client,
		maxTrials:  DefaultMaxTrials,
		memorySize: DefaultMemorySize,
	}

	for _, opt := range options {
		opt(s)
	}

	// Set default evaluator if not provided
	if s.evaluator == nil {
		s.evaluator = NewLLMEvaluator(client)
	}

	return s
}

// Init initializes the strategy with initial inputs.
// This is called once when Agent.Execute is invoked.
func (s *Strategy) Init(ctx context.Context, inputs []gollem.Input) error {
	s.currentTrial = 0
	s.trials = nil
	s.memory = newMemory(s.memorySize)
	return nil
}

// Tools returns additional tools provided by this strategy.
// Reflexion strategy does not provide additional tools.
func (s *Strategy) Tools(ctx context.Context) ([]gollem.Tool, error) {
	return []gollem.Tool{}, nil
}

// Handle determines the next input for the LLM based on the current state.
// It manages the trial loop: execution, evaluation, reflection, and retry.
func (s *Strategy) Handle(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
	// Phase 0: Initialize first trial
	if state.Iteration == 0 {
		return s.startTrial(ctx, state)
	}

	// Phase 1: Pass through during trial execution
	if s.isTrialInProgress() {
		// Check for trial completion
		if s.isTrialComplete(state) {
			return s.completeTrial(ctx, state)
		}
		// Continue execution (pass through tool responses, etc.)
		return state.NextInput, nil, nil
	}

	// Unexpected state
	return nil, nil, goerr.New("unexpected state in Handle")
}
