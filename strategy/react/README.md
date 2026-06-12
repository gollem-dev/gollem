# ReAct Strategy

ReAct (Reasoning and Acting) is a strategy for gollem that implements the framework described in the paper "[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)".

## Overview

The ReAct strategy alternates between three phases to solve complex problems:

1. **Thought**: The LLM reasons about the current situation and decides what to do next
2. **Action**: The LLM takes an action (calls a tool or provides a final response)
3. **Observation**: The results of the action are observed and fed back into the next thought cycle

This cycle continues until the LLM has gathered sufficient information to provide a final answer.

## Features

- **Explicit TAO Cycle**: Clearly structured Thought-Action-Observation phases
- **Loop Detection**: Prevents infinite loops by detecting repeated actions
- **Error Handling**: Gracefully handles tool execution errors with retry logic
- **Safety Limits**: Maximum iteration count to prevent runaway execution
- **Trace Export**: Complete execution history can be exported for debugging and analysis
- **Customizable Prompts**: All prompts can be customized to fit your use case

## Basic Usage

```go
package main

import (
    "context"
    "fmt"

    "github.com/gollem-dev/gollem"
    "github.com/gollem-dev/gollem/llm/openai"
    "github.com/gollem-dev/gollem/strategy/react"
)

func main() {
    ctx := context.Background()

    // Create LLM client
    llmClient, err := openai.New(ctx, "your-api-key")
    if err != nil {
        panic(err)
    }

    // Create ReAct strategy
    strategy := react.New(llmClient,
        react.WithMaxIterations(20),
        react.WithMaxRepeatedActions(3),
    )

    // Create agent with ReAct strategy
    agent := gollem.New(llmClient, gollem.WithStrategy(strategy))

    // Execute
    response, err := agent.Execute(ctx, gollem.Text("What is the weather in Tokyo?"))
    if err != nil {
        panic(err)
    }

    fmt.Println(response)

    // Export trace for analysis
    trace := strategy.ExportTrace()
    fmt.Printf("Total iterations: %d\n", trace.Summary.TotalIterations)
    fmt.Printf("Tool calls: %d\n", trace.Summary.ToolCallsCount)
}
```

## Configuration Options

### WithMaxIterations

Sets the maximum number of iterations before forcibly terminating. Default is 20.

```go
strategy := react.New(llmClient,
    react.WithMaxIterations(10),
)
```

### WithMaxRepeatedActions

Sets how many times the same action can be repeated before detecting a loop. Default is 3.

```go
strategy := react.New(llmClient,
    react.WithMaxRepeatedActions(5),
)
```

### WithThoughtPrompt

Customizes the prompt used to encourage reasoning.

```go
strategy := react.New(llmClient,
    react.WithThoughtPrompt("Think carefully about the next step:"),
)
```

### WithObservationPrompt

Customizes the prompt template for observations. Should contain two `%s` placeholders for tool name and result.

```go
strategy := react.New(llmClient,
    react.WithObservationPrompt("Tool %s returned: %s\nWhat should we do next?"),
)
```

### WithFewShotExamples

Enables few-shot learning with provided examples (planned for future release).

```go
examples := []react.FewShotExample{
    {
        Question: "What is 2+2?",
        Thought: "I need to calculate 2+2",
        Action: "calculator(2+2)",
        Observation: "4",
        Answer: "The answer is 4",
    },
}

strategy := react.New(llmClient,
    react.WithFewShotExamples(examples),
)
```

## Trace Export

The ReAct strategy records the complete execution trace, which can be exported for debugging and analysis.

```go
// Export trace
trace := strategy.ExportTrace()

// Access summary
fmt.Printf("Total iterations: %d\n", trace.Summary.TotalIterations)
fmt.Printf("Success rate: %.2f\n", trace.Summary.SuccessRate)
fmt.Printf("Duration: %s\n", trace.Summary.Duration)

// Access individual entries
for _, entry := range trace.Entries {
    if entry.Thought != nil {
        fmt.Printf("Thought: %s\n", entry.Thought.Content)
    }
    if entry.Action != nil {
        fmt.Printf("Action: %s\n", entry.Action.Type)
        if entry.Action.Type == react.ActionTypeToolCall {
            for _, call := range entry.Action.ToolCalls {
                fmt.Printf("  - Tool Call: %s with args %v\n", call.Name, call.Arguments)
            }
        } else if entry.Action.Type == react.ActionTypeRespond {
            fmt.Printf("  - Response: %s\n", entry.Action.Response)
        }
    }
    if entry.Observation != nil {
        fmt.Printf("Observation: success=%v\n", entry.Observation.Success)
        if !entry.Observation.Success {
            fmt.Printf("  - Error: %s\n", entry.Observation.Error)
        }
    }
}

// Export as JSON
jsonData, err := strategy.ExportTraceJSON()
if err != nil {
    panic(err)
}
fmt.Println(string(jsonData))
```

## Safety Features

### Loop Detection

The strategy detects when the same action is repeated multiple times and automatically terminates to prevent infinite loops.

```go
strategy := react.New(llmClient,
    react.WithMaxRepeatedActions(3), // Stop after 3 repeated actions
)
```

### Maximum Iterations

A hard limit on the number of iterations prevents runaway execution.

```go
strategy := react.New(llmClient,
    react.WithMaxIterations(20), // Stop after 20 iterations
)
```

### Error Handling

Tool execution errors are gracefully handled:
- Errors are recorded in the observation
- The LLM is informed of the error and can retry or try a different approach
- After 3 consecutive errors, execution is terminated

## Best Practices

1. **Set appropriate iteration limits**: Too low and complex tasks may not complete; too high and you risk excessive API costs.

2. **Customize prompts for your domain**: The default prompts are general-purpose. Customizing them can improve performance for specific use cases.

3. **Monitor the trace**: Use `ExportTrace()` to understand how the LLM is reasoning and identify areas for improvement.

4. **Provide clear tool descriptions**: The ReAct strategy works best when tools have clear, concise descriptions that explain what they do.

5. **Handle errors gracefully**: Design your tools to return meaningful error messages that the LLM can understand and act upon.

## Troubleshooting

### The agent keeps repeating the same action

Reduce `WithMaxRepeatedActions` to detect loops earlier, or improve your tool descriptions so the LLM understands when a tool has already been tried.

### The agent runs out of iterations before completing

Increase `WithMaxIterations`, or break down your problem into smaller sub-problems.

### Tool errors are not being handled well

Ensure your tools return descriptive error messages. The LLM needs to understand what went wrong to try a different approach.

## References

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [gollem documentation](https://github.com/gollem-dev/gollem)
