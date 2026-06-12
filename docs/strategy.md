# Strategy Pattern for Agent Behavior

gollem uses the Strategy pattern to customize how agents process tasks and make decisions. Strategies control the core execution logic, from simple request-response to complex planning workflows.

## Built-in Strategies

### Default Strategy

Simple request-response pattern. Used automatically when no strategy is specified:

```go
agent := gollem.New(client,
	gollem.WithTools(&MyTool{}),
)
```

### React Strategy

ReAct (Reasoning + Acting) pattern with step-by-step reasoning:

```go
import "github.com/gollem-dev/gollem/strategy/react"

strategy := react.New(client)
agent := gollem.New(client,
	gollem.WithStrategy(strategy),
	gollem.WithTools(&MyTool{}),
)
```

### Plan & Execute Strategy

Goal-oriented task planning and execution with context-aware planning:

```go
import "github.com/gollem-dev/gollem/strategy/planexec"

strategy := planexec.New(client)
agent := gollem.New(client,
	gollem.WithStrategy(strategy),
	gollem.WithSystemPrompt("You are an expert data analyst. All outputs must be HIPAA compliant."),
	gollem.WithHistory(history), // Use conversation history
	gollem.WithTools(&SearchTool{}, &AnalysisTool{}),
)
```

The Plan & Execute strategy uses a three-phase approach with context embedding:
- **Planning**: Creates task breakdown with system prompt and history context. Important constraints and requirements are embedded into the Plan structure (`context_summary` and `constraints` fields), making the plan self-contained.
- **Execution**: Executes tasks sequentially using the main session with full context (system prompt and history).
- **Reflection**: After each task completion, evaluates results using ONLY the Plan's embedded information (goal, context summary, constraints) without accessing the original system prompt or history. This ensures consistent evaluation criteria and enables stateless reflection.

### External Plan Generation

You can generate and reuse plans separately from execution:

```go
import "github.com/gollem-dev/gollem/strategy/planexec"

// Generate a plan separately
plan, err := planexec.GeneratePlan(ctx, client,
	[]gollem.Input{gollem.Text("Analyze security logs")},
	tools,                       // Available tools
	"Focus on OWASP Top 10",    // System prompt
	nil,                         // History (optional)
)

// Save plan for later or review
planData, _ := json.Marshal(plan)
savePlan(planData)

// Later: load and execute with pre-generated plan
var savedPlan *planexec.Plan
json.Unmarshal(planData, &savedPlan)

strategy := planexec.New(client, planexec.WithPlan(savedPlan))
agent := gollem.New(client, gollem.WithStrategy(strategy), gollem.WithTools(tools...))
resp, err := agent.Execute(ctx, gollem.Text("Analyze security logs"))
```

This enables use cases like:
- **Plan Review**: Generate plan, review tasks, then execute
- **Plan Caching**: Reuse plans for similar requests
- **Plan Modification**: Adjust tasks before execution
- **Parallel Planning**: Generate plans with one model, execute with another

## Custom Strategy Implementation

Implement the `Strategy` interface for custom agent behavior:

```go
type Strategy interface {
	Init(ctx context.Context, inputs []Input) error
	Handle(ctx context.Context, state *StrategyState) ([]Input, *ExecuteResponse, error)
	Tools(ctx context.Context) ([]Tool, error)
}
```

**Example: Custom Strategy**
```go
type myStrategy struct {
	client gollem.LLMClient
}

func (s *myStrategy) Init(ctx context.Context, inputs []Input) error {
	// Initialize strategy state
	return nil
}

func (s *myStrategy) Handle(ctx context.Context, state *StrategyState) ([]Input, *ExecuteResponse, error) {
	// Custom decision logic
	resp, err := state.Generate(ctx, state.Inputs)
	if err != nil {
		return nil, nil, err
	}

	// Return next inputs and optional completion response
	return nil, &gollem.ExecuteResponse{Message: resp.Texts[0]}, nil
}

func (s *myStrategy) Tools(ctx context.Context) ([]Tool, error) {
	// Return available tools for this strategy
	return []Tool{}, nil
}

// Use custom strategy
agent := gollem.New(client,
	gollem.WithStrategy(&myStrategy{client: client}),
)
```

**Key Features:**
- **Pluggable Architecture**: Swap strategies without changing agent code
- **Built-in Patterns**: ReAct, Plan & Execute, or default simple execution
- **Custom Logic**: Implement your own decision-making algorithms
- **State Management**: Full access to conversation state and history

## Next Steps

- Learn about [middleware](middleware.md) for monitoring and control
- Explore [Plan Mode](plan-mode.md) for detailed planning workflows
- Review [tracing](tracing.md) for strategy execution observability
