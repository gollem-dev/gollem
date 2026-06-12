package planexec

import (
	"context"

	"github.com/gollem-dev/gollem"
)

// TaskState represents the current state of a task
type TaskState string

const (
	TaskStatePending    TaskState = "pending"
	TaskStateInProgress TaskState = "in_progress"
	TaskStateCompleted  TaskState = "completed"
	TaskStateSkipped    TaskState = "skipped" // Task was skipped (already executed or no longer needed)

	// DefaultMaxIterations is the default maximum number of task execution iterations
	DefaultMaxIterations = 32
)

// Task represents an executable task in the plan
type Task struct {
	ID          string // Unique identifier for the task
	Description string
	State       TaskState
	Result      string
}

// Plan represents the execution plan with tasks
type Plan struct {
	// User's original input question (e.g., "Investigate X", "Analyze the data")
	UserQuestion string

	// User's true intent - what they want to know (e.g., "Want to know the investigation result", "Understand the data patterns")
	// This is result-oriented, not process-oriented
	UserIntent string

	// Specific goal to accomplish the intent (e.g., "Find where authentication happens")
	Goal string

	Tasks          []Task
	DirectResponse string // Used when no plan is needed

	// Context embedded from system prompt and history for self-contained evaluation
	// This information is used during reflection to evaluate task completion
	// without needing access to the original system prompt or conversation history
	ContextSummary string // Summary of relevant context from system prompt and history
	Constraints    string // Key constraints and requirements (e.g., "HIPAA compliance required")
}

// PlanExecuteHooks provides hook points for plan lifecycle events
type PlanExecuteHooks interface {
	OnPlanCreated(ctx context.Context, plan *Plan) error
	OnPlanUpdated(ctx context.Context, plan *Plan) error
	OnTaskDone(ctx context.Context, plan *Plan, task *Task) error
}

// Strategy implements the gollem.Strategy interface for plan-and-execute approach
type Strategy struct {
	client        gollem.LLMClient
	middleware    []gollem.ContentBlockMiddleware
	hooks         PlanExecuteHooks
	maxIterations int

	// Runtime state
	plan               *Plan
	planProvidedByUser bool // true if plan was provided via WithPlan option
	planCreatedHookRan bool // true if OnPlanCreated hook has been called
	currentTask        *Task
	waitingForTask     bool
	taskIterationCount int // Counts completed tasks

	// Temporary storage for tool execution results
	// When NextInput contains tool results, save them here before passing to LLM
	pendingToolResults []gollem.Input
}

// Option is a functional option for configuring Strategy
type Option func(*Strategy)
