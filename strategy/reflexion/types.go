package reflexion

import (
	"context"
	"time"

	"github.com/gollem-dev/gollem"
)

// Trajectory captures the execution trace of a trial.
// It contains the complete execution history including user inputs, LLM responses,
// tool executions, and the final response.
type Trajectory struct {
	TrialNum      int             // Trial number (1-indexed)
	UserInputs    []gollem.Input  // Initial user inputs for the task
	History       *gollem.History // Complete conversation history
	FinalResponse []string        // Final response texts from LLM
	StartTime     time.Time       // Trial start time
	EndTime       time.Time       // Trial end time
}

// EvaluationResult represents the result of evaluating a trajectory.
// It indicates whether the trial was successful and optionally provides
// additional feedback and scoring information.
type EvaluationResult struct {
	Success  bool    // Whether the trial successfully completed the task
	Score    float64 // Optional score (0.0-1.0), 0 if not used
	Feedback string  // Optional feedback or explanation
}

// Evaluator evaluates a trajectory to determine if it successfully completed the task.
// Users can provide custom evaluation logic as a function.
//
// Example:
//
//	evaluator := reflexion.Evaluator(func(ctx context.Context, t *reflexion.Trajectory) (*reflexion.EvaluationResult, error) {
//	    // Custom evaluation logic
//	    if containsExpectedOutput(t.FinalResponse) {
//	        return &reflexion.EvaluationResult{Success: true}, nil
//	    }
//	    return &reflexion.EvaluationResult{
//	        Success: false,
//	        Feedback: "Output does not match expected result",
//	    }, nil
//	})
type Evaluator func(context.Context, *Trajectory) (*EvaluationResult, error)

// Hooks provides lifecycle hooks for observing the Reflexion strategy's execution.
// All methods are optional - implement only the hooks you need.
type Hooks interface {
	// OnTrialStart is called when a new trial begins.
	OnTrialStart(ctx context.Context, trialNum int) error

	// OnTrialEnd is called when a trial completes (success or failure).
	OnTrialEnd(ctx context.Context, trialNum int, evaluation *EvaluationResult) error

	// OnReflectionGenerated is called when a reflection is generated after a failed trial.
	OnReflectionGenerated(ctx context.Context, trialNum int, reflection string) error
}
