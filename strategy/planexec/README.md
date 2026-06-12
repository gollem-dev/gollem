# Plan & Execute Strategy

The Plan & Execute strategy implements a goal-oriented approach for complex, multi-step tasks by separating planning from execution. This strategy is ideal for tasks that require breaking down high-level goals into actionable steps, executing them with available tools, and adapting based on results.

## Overview

The Plan & Execute strategy follows a three-phase cycle:

1. **Planning**: Analyze user intent and create a structured task list
2. **Execution**: Execute each task sequentially using available tools
3. **Reflection**: After each task, evaluate progress and adapt the plan if needed

This approach provides:
- **Clear Intent**: Separates strategic planning from tactical execution
- **Adaptability**: Plans evolve based on real execution results
- **Context Preservation**: Embeds important context into the plan structure
- **Stateless Reflection**: Reflection uses only plan-embedded information

## Features

- **Context-Aware Planning**: Incorporates system prompt and conversation history into planning
- **Adaptive Task Management**: Add, update, or skip tasks based on execution results
- **Lifecycle Hooks**: Monitor and control plan execution at key points
- **External Plan Generation**: Generate and reuse plans separately from execution
- **Plan Serialization**: Save and restore plans for async or distributed workflows
- **Middleware Support**: Apply custom middleware to planning sessions
- **Iteration Limits**: Prevent infinite loops with configurable task iteration limits

## Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/gollem-dev/gollem"
    "github.com/gollem-dev/gollem/llm/openai"
    "github.com/gollem-dev/gollem/strategy/planexec"
)

func main() {
    ctx := context.Background()

    // Create LLM client
    client, err := openai.New(ctx, os.Getenv("OPENAI_API_KEY"))
    if err != nil {
        panic(err)
    }

    // Create Plan & Execute strategy
    strategy := planexec.New(client,
        planexec.WithMaxIterations(32),
        planexec.WithHooks(&myHooks{}),
    )

    // Create agent with strategy
    agent := gollem.New(client,
        gollem.WithStrategy(strategy),
        gollem.WithTools(&SearchTool{}, &AnalysisTool{}),
        gollem.WithSystemPrompt("You are a data analyst."),
    )

    // Execute
    response, err := agent.Execute(ctx, gollem.Text("Analyze user behavior patterns"))
    if err != nil {
        panic(err)
    }

    fmt.Println(response.Texts)
}
```

## External Plan Generation

Generate plans separately from execution for review, caching, or modification:

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "os"

    "github.com/gollem-dev/gollem"
    "github.com/gollem-dev/gollem/llm/openai"
    "github.com/gollem-dev/gollem/strategy/planexec"
)

func main() {
    ctx := context.Background()
    client, _ := openai.New(ctx, os.Getenv("OPENAI_API_KEY"))

    // Define available tools
    tools := []gollem.Tool{&SearchTool{}, &AnalysisTool{}}

    // Generate plan separately
    plan, err := planexec.GeneratePlan(ctx, client,
        []gollem.Input{gollem.Text("Analyze security logs for vulnerabilities")},
        tools,                               // Available tools
        "Focus on OWASP Top 10 vulnerabilities", // System prompt
        nil,                                 // History (optional)
    )
    if err != nil {
        panic(err)
    }

    // Review plan tasks
    fmt.Printf("Plan created with %d tasks:\n", len(plan.Tasks))
    for i, task := range plan.Tasks {
        fmt.Printf("%d. %s\n", i+1, task.Description)
    }

    // Save plan for later (e.g., to database)
    planData, _ := json.Marshal(plan)
    saveToDatabase(planData)

    // Later: load and execute with pre-generated plan
    var savedPlan *planexec.Plan
    loadedData := loadFromDatabase()
    json.Unmarshal(loadedData, &savedPlan)

    // Create strategy with existing plan
    strategy := planexec.New(client, planexec.WithPlan(savedPlan))
    agent := gollem.New(client,
        gollem.WithStrategy(strategy),
        gollem.WithTools(tools...),
    )

    // Execute with pre-generated plan
    response, err := agent.Execute(ctx, gollem.Text("Analyze security logs"))
    if err != nil {
        panic(err)
    }

    fmt.Println(response.Texts)
}
```

### Use Cases for External Plans

1. **Plan Review**: Generate plan, review tasks with humans, then execute
2. **Plan Caching**: Reuse plans for similar requests to save planning costs
3. **Plan Modification**: Programmatically adjust tasks before execution
4. **Parallel Planning**: Generate plans with one model, execute with another
5. **Async Workflows**: Generate plan in one service, execute in another

## Configuration Options

### WithMaxIterations

Sets the maximum number of iterations per task execution. Default is 32.

```go
strategy := planexec.New(client,
    planexec.WithMaxIterations(16), // Limit to 16 iterations per task
)
```

### WithHooks

Provides lifecycle hooks for monitoring and controlling execution.

```go
type myHooks struct{}

func (h *myHooks) OnPlanCreated(ctx context.Context, plan *planexec.Plan) error {
    fmt.Printf("Plan created: %s\n", plan.Goal)
    return nil
}

func (h *myHooks) OnPlanUpdated(ctx context.Context, plan *planexec.Plan) error {
    fmt.Println("Plan updated with new or modified tasks")
    return nil
}

func (h *myHooks) OnTaskDone(ctx context.Context, plan *planexec.Plan, task *planexec.Task) error {
    fmt.Printf("Task completed: %s (state: %s)\n", task.Description, task.State)
    return nil
}

strategy := planexec.New(client, planexec.WithHooks(&myHooks{}))
```

### WithMiddleware

Applies middleware to planning and reflection sessions.

```go
loggingMiddleware := func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
    return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
        resp, err := next(ctx, req)
        if err == nil {
            log.Printf("Planning response: %v tokens", resp.OutputToken)
        }
        return resp, err
    }
}

strategy := planexec.New(client,
    planexec.WithMiddleware(loggingMiddleware),
)
```

### WithPlan

Uses a pre-generated plan instead of creating a new one.

```go
// Generate plan
plan, _ := planexec.GeneratePlan(ctx, client, inputs, options...)

// Use pre-generated plan
strategy := planexec.New(client, planexec.WithPlan(plan))
```

## GeneratePlan Function Signature

```go
func GeneratePlan(
    ctx context.Context,
    client gollem.LLMClient,
    inputs []gollem.Input,
    tools []gollem.Tool,        // Available tools (can be nil)
    systemPrompt string,         // System prompt (can be empty)
    history *gollem.History,     // Conversation history (can be nil)
) (*Plan, error)
```

### Parameters

- **ctx**: Context for cancellation and timeout
- **client**: LLM client to use for plan generation
- **inputs**: User inputs to analyze and plan for
- **tools**: Tools available for task execution (used for planning context)
- **systemPrompt**: System prompt to guide planning behavior
- **history**: Conversation history for context-aware planning

### Example

```go
// Simple plan generation
plan, err := planexec.GeneratePlan(ctx, client,
    []gollem.Input{gollem.Text("Analyze security logs")},
    nil,  // No tools
    "",   // No system prompt
    nil,  // No history
)

// Plan with full context
plan, err := planexec.GeneratePlan(ctx, client,
    []gollem.Input{gollem.Text("Analyze security logs")},
    []gollem.Tool{&SearchTool{}, &AnalysisTool{}},
    "You are a security expert. Focus on OWASP vulnerabilities.",
    conversationHistory,
)
```

## Lifecycle Hooks

### OnPlanCreated

Called once when a plan is created (including pre-generated plans).

```go
func (h *myHooks) OnPlanCreated(ctx context.Context, plan *planexec.Plan) error {
    log.Printf("Created plan: %s", plan.Goal)
    log.Printf("Tasks: %d", len(plan.Tasks))
    return nil
}
```

### OnPlanUpdated

Called when tasks are added or modified during reflection.

```go
func (h *myHooks) OnPlanUpdated(ctx context.Context, plan *planexec.Plan) error {
    log.Println("Plan updated with new tasks")
    return nil
}
```

### OnTaskDone

Called after each task completes.

```go
func (h *myHooks) OnTaskDone(ctx context.Context, plan *planexec.Plan, task *planexec.Task) error {
    if task.State == planexec.TaskStateCompleted {
        log.Printf("✅ %s", task.Description)
    } else if task.State == planexec.TaskStateSkipped {
        log.Printf("⏭️  %s", task.Description)
    }
    return nil
}
```

## Plan Structure

```go
type Plan struct {
    UserQuestion   string  // Original user input
    UserIntent     string  // What user wants to know (result-oriented)
    Goal           string  // Specific goal to accomplish
    Tasks          []Task  // Executable tasks
    DirectResponse string  // Used when no tasks needed
    ContextSummary string  // Embedded context from system prompt/history
    Constraints    string  // Key requirements (e.g., "HIPAA compliance")
}

type Task struct {
    ID          string     // Unique identifier
    Description string     // Task description
    State       TaskState  // pending, in_progress, completed, skipped
    Result      string     // Execution result
}
```

## How It Works

### 1. Planning Phase

The strategy creates a plan by:
- Analyzing user input, system prompt, and conversation history
- Extracting user intent and constraints
- Breaking down the goal into concrete tasks
- Embedding important context into the plan structure

```json
{
  "needs_plan": true,
  "user_intent": "Want to identify security vulnerabilities",
  "goal": "Analyze logs for OWASP Top 10 vulnerabilities",
  "context_summary": "Security audit context with HIPAA compliance",
  "constraints": "Must follow HIPAA guidelines",
  "tasks": [
    {"description": "Scan authentication logs for injection attempts"},
    {"description": "Check for broken access control"},
    {"description": "Generate vulnerability report"}
  ]
}
```

### 2. Execution Phase

Each task is executed sequentially:
- Uses main session with full context (tools, system prompt, history)
- LLM decides which tools to call
- Task results are captured for reflection

### 3. Reflection Phase

After each task completion:
- Evaluates progress using plan's embedded context (goal, constraints)
- Decides whether to continue, skip tasks, or add new ones
- Updates plan based on discoveries
- Does NOT access original system prompt or history (enables stateless reflection)

```json
{
  "new_tasks": ["Prioritize vulnerabilities by severity"],
  "updated_tasks": [],
  "reason": "Found critical issues requiring prioritization"
}
```

## Best Practices

### 1. Provide Clear System Prompts

```go
// Good: Specific domain guidance
agent := gollem.New(client,
    gollem.WithSystemPrompt("You are a security analyst. Focus on OWASP Top 10. All findings must be HIPAA compliant."),
)

// Less effective: Too generic
agent := gollem.New(client,
    gollem.WithSystemPrompt("You are helpful."),
)
```

### 2. Use Hooks for Monitoring

```go
strategy := planexec.New(client,
    planexec.WithHooks(&progressTracker{
        startTime: time.Now(),
    }),
)
```

### 3. Set Appropriate Iteration Limits

```go
// For quick tasks
strategy := planexec.New(client, planexec.WithMaxIterations(10))

// For complex analysis
strategy := planexec.New(client, planexec.WithMaxIterations(50))
```

### 4. Review Generated Plans

```go
// Generate and review before executing
plan, _ := planexec.GeneratePlan(ctx, client, inputs, options...)

// Review tasks
for _, task := range plan.Tasks {
    fmt.Println(task.Description)
}

// Approve and execute
if approved {
    strategy := planexec.New(client, planexec.WithPlan(plan))
    // ... execute
}
```

## Troubleshooting

### Plan creates too many/few tasks

Adjust system prompt to guide task granularity when generating the plan:

```go
// When using GeneratePlan directly
plan, err := planexec.GeneratePlan(ctx, client, inputs, tools,
    `Break down the goal into 3-5 concrete, actionable tasks.
Each task should be completable with available tools.`,
    history,
)

// When using strategy with agent
agent := gollem.New(client,
    gollem.WithStrategy(planexec.New(client)),
    gollem.WithSystemPrompt(`Break down the goal into 3-5 concrete, actionable tasks.
Each task should be completable with available tools.`),
)
```

### Reflection adds unnecessary tasks

The reflection is too aggressive. This is usually because the goal or constraints are unclear. Improve the system prompt:

```go
agent := gollem.New(client,
    gollem.WithStrategy(planexec.New(client)),
    gollem.WithSystemPrompt(`Focus on the core goal. Only add tasks if absolutely necessary to achieve the objective.`),
)
```

### Tasks hit iteration limit

Increase max iterations or simplify tasks:

```go
planexec.WithMaxIterations(64) // Increase limit
```

## Examples

See [examples/](../../examples/) directory for complete examples:
- Basic plan execution
- External plan generation and reuse
- Progress monitoring with hooks
- Custom middleware integration

## References

- [gollem documentation](https://github.com/gollem-dev/gollem)
- [Plan & Execute Pattern](https://arxiv.org/abs/2305.04091)
