# Reflexion Strategy

The Reflexion strategy implements the Reflexion framework from the NeurIPS 2023 paper: ["Reflexion: Language Agents with Verbal Reinforcement Learning"](https://arxiv.org/abs/2303.11366).

## Overview

Reflexion enables language agents to learn from their mistakes through verbal feedback and self-reflection. Unlike traditional reinforcement learning, Reflexion uses natural language feedback stored in episodic memory to improve performance across multiple trials.

## Key Components

### Actor
The LLM agent that executes tasks using the available tools and context.

### Evaluator
Determines whether a trial successfully completed the task. You can use:
- **LLMEvaluator** (default): Uses an LLM to evaluate success
- **Custom Evaluator**: Provide a custom function of type `Evaluator`

### Self-Reflection
Generates verbal feedback after failed trials, which is stored in episodic memory and used to guide future attempts.

### Episodic Memory
A bounded FIFO buffer storing reflections from past trials (default: 3 most recent).

## Usage

### Basic Example

```go
import (
    "github.com/gollem-dev/gollem"
    "github.com/gollem-dev/gollem/llm/openai"
    "github.com/gollem-dev/gollem/strategy/reflexion"
)

// Create LLM client
client, err := openai.New(ctx, apiKey)
if err != nil {
    // Handle error
}

// Create Reflexion strategy with default LLM evaluator
strategy := reflexion.New(client,
    reflexion.WithMaxTrials(3),
    reflexion.WithMemorySize(3),
)

// Create agent
agent := gollem.New(client, gollem.WithStrategy(strategy))

// Execute task
response, err := agent.Execute(ctx, gollem.Text("Your task here..."))
if err != nil {
    // Handle error
}
```

### Custom Evaluator

```go
// Simple function-based evaluator
evaluator := reflexion.Evaluator(func(ctx context.Context, t *reflexion.Trajectory) (*reflexion.EvaluationResult, error) {
    response := strings.Join(t.FinalResponse, " ")

    // Check if response meets your criteria
    if containsExpectedOutput(response) {
        return &reflexion.EvaluationResult{
            Success: true,
            Score:   1.0,
        }, nil
    }

    return &reflexion.EvaluationResult{
        Success:  false,
        Feedback: "Response missing expected content",
    }, nil
})

strategy := reflexion.New(client,
    reflexion.WithEvaluator(evaluator),
)
```

### Using Hooks

Hooks allow you to observe and log the Reflexion process:

```go
type myHooks struct{}

func (h *myHooks) OnTrialStart(ctx context.Context, trialNum int) error {
    log.Printf("Starting trial %d", trialNum)
    return nil
}

func (h *myHooks) OnTrialEnd(ctx context.Context, trialNum int, evaluation *reflexion.EvaluationResult) error {
    log.Printf("Trial %d ended - Success: %v", trialNum, evaluation.Success)
    return nil
}

func (h *myHooks) OnReflectionGenerated(ctx context.Context, trialNum int, reflection string) error {
    log.Printf("Reflection: %s", reflection)
    return nil
}

strategy := reflexion.New(client,
    reflexion.WithHooks(&myHooks{}),
)
```

## Configuration Options

### WithMaxTrials(n int)
Sets the maximum number of trials (default: 3).

```go
reflexion.WithMaxTrials(5)
```

### WithMemorySize(n int)
Sets the size of episodic memory (default: 3).

```go
reflexion.WithMemorySize(5)
```

### WithEvaluator(evaluator Evaluator)
Sets a custom evaluator (default: LLMEvaluator).

```go
reflexion.WithEvaluator(myEvaluator)
```

### WithHooks(hooks Hooks)
Sets lifecycle hooks for observability.

```go
reflexion.WithHooks(myHooks)
```

## How It Works

1. **Trial Execution**: The agent executes the task with the current context
2. **Evaluation**: The evaluator determines if the trial succeeded
3. **Success Path**: If successful, return the final response
4. **Failure Path**:
   - Generate a reflection on what went wrong
   - Store reflection in episodic memory
   - Start next trial with reflections as additional context
5. **Retry**: Continue until success or max trials reached

## Examples

See [examples/reflexion/](../../examples/reflexion/) for a complete working example.

## Comparison with Other Strategies

| Strategy | Use Case | Learning | Trials |
|----------|----------|----------|--------|
| **Simple** | Basic tasks | No | Single |
| **ReAct** | Step-by-step reasoning | No | Single |
| **Reflexion** | Tasks requiring improvement | Yes (episodic memory) | Multiple |
| **PlanExec** | Goal-oriented planning | No | Single |

## References

- [Reflexion Paper (NeurIPS 2023)](https://arxiv.org/abs/2303.11366)
- [Official Implementation](https://github.com/noahshinn024/reflexion)
