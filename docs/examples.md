# Practical Examples

This guide provides practical examples of using gollem in various scenarios.

## Basic Agent Usage

Here's a simple example using the recommended `Execute` method:

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/gollem-dev/gollem"
    "github.com/gollem-dev/gollem/llm/openai"
)

func main() {
    ctx := context.Background()

    // Create OpenAI client
    client, err := openai.New(ctx, os.Getenv("OPENAI_API_KEY"))
    if err != nil {
        panic(err)
    }

    // Create agent with automatic session management
    agent := gollem.New(client,
        gollem.WithSystemPrompt("You are a helpful assistant."),
        gollem.WithContentBlockMiddleware(func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
            return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
                resp, err := next(ctx, req)
                if err == nil && len(resp.Texts) > 0 {
                    for _, text := range resp.Texts {
                        fmt.Printf("🤖 %s\n", text)
                    }
                }
                return resp, err
            }
        }),
    )

    // Execute multiple interactions - history managed automatically
    err = agent.Execute(ctx, "Hello, I'm working on a Go project.")
    if err != nil {
        panic(err)
    }

    err = agent.Execute(ctx, "Can you help me with error handling best practices?")
    if err != nil {
        panic(err)
    }
}
```

## Calculator Tool

A more complex example using a calculator tool:

```go
type CalculatorTool struct{}

func (t *CalculatorTool) Spec() gollem.ToolSpec {
    return gollem.ToolSpec{
        Name:        "calculator",
        Description: "Performs basic arithmetic operations",
        Parameters: map[string]*gollem.Parameter{
            "operation": {
                Type:        gollem.TypeString,
                Description: "The operation to perform (add, subtract, multiply, divide)",
            },
            "a": {
                Type:        gollem.TypeNumber,
                Description: "First number",
            },
            "b": {
                Type:        gollem.TypeNumber,
                Description: "Second number",
            },
        },
        Required: []string{"operation", "a", "b"},
    }
}

func (t *CalculatorTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
    // Validate operation
    op, ok := args["operation"].(string)
    if !ok {
        return nil, errors.New("operation must be a string")
    }
    if op != "add" && op != "subtract" && op != "multiply" && op != "divide" {
        return nil, errors.New("invalid operation: must be one of add, subtract, multiply, divide")
    }

    // Validate first number
    a, ok := args["a"].(float64)
    if !ok {
        return nil, errors.New("first number must be a number")
    }

    // Validate second number
    b, ok := args["b"].(float64)
    if !ok {
        return nil, errors.New("second number must be a number")
    }

    // Validate division by zero
    if op == "divide" && b == 0 {
        return nil, errors.New("division by zero is not allowed")
    }

    var result float64
    switch op {
    case "add":
        result = a + b
    case "subtract":
        result = a - b
    case "multiply":
        result = a * b
    case "divide":
        result = a / b
    }

    return map[string]any{
        "result": result,
    }, nil
}

// Usage example
func main() {
    ctx := context.Background()
    client, _ := openai.New(ctx, os.Getenv("OPENAI_API_KEY"))

    agent := gollem.New(client,
        gollem.WithTools(&CalculatorTool{}),
        gollem.WithContentBlockMiddleware(func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
            return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
                resp, err := next(ctx, req)
                if err == nil && len(resp.Texts) > 0 {
                    for _, text := range resp.Texts {
                        fmt.Printf("🤖 %s\n", text)
                    }
                }
                return resp, err
            }
        }),
        gollem.WithToolMiddleware(func(next gollem.ToolHandler) gollem.ToolHandler {
            return func(ctx context.Context, req *gollem.ToolExecRequest) (*gollem.ToolExecResponse, error) {
                fmt.Printf("⚡ Executing: %s\n", req.Tool.Name)
                return next(ctx, req)
            }
        }),
    )

    err := agent.Execute(ctx, "Calculate 15 + 27 and then multiply the result by 3")
    if err != nil {
        panic(err)
    }
}
```

## Weather Tool with MCP

An example of a weather tool using MCP:

```go
type WeatherTool struct{}

func (t *WeatherTool) Spec() gollem.ToolSpec {
    return gollem.ToolSpec{
        Name:        "weather",
        Description: "Gets weather information for a location",
        Parameters: map[string]*gollem.Parameter{
            "location": {
                Type:        gollem.TypeString,
                Description: "City name or coordinates",
            },
        },
        Required: []string{"location"},
    }
}

func (t *WeatherTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
    location, ok := args["location"].(string)
    if !ok {
        return nil, fmt.Errorf("location must be a string")
    }
    
    // Simulate weather API call
    weatherData := map[string]any{
        "location":    location,
        "temperature": 25.5,
        "condition":   "sunny",
        "humidity":    60,
        "wind_speed":  10.2,
    }
    
    return weatherData, nil
}
```

## Comprehensive Middleware Usage

Example showing middleware for monitoring and control:

```go
func createMonitoredAgent(client gollem.LLMClient) *gollem.Agent {
    return gollem.New(client,
        // Content block middleware for real-time output
        gollem.WithContentBlockMiddleware(func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
            return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
                resp, err := next(ctx, req)
                if err == nil && len(resp.Texts) > 0 {
                    for _, text := range resp.Texts {
                        fmt.Printf("🤖 LLM: %s\n", text)
                        // Could stream to WebSocket, save to database, etc.
                    }
                }
                return resp, err
            }
        }),

        // Tool middleware for monitoring, access control, and error handling
        gollem.WithToolMiddleware(func(next gollem.ToolHandler) gollem.ToolHandler {
            return func(ctx context.Context, req *gollem.ToolExecRequest) (*gollem.ToolExecResponse, error) {
                fmt.Printf("⚡ Executing tool: %s with args: %v\n", req.Tool.Name, req.Tool.Arguments)

                // Example: Implement rate limiting
                if isRateLimited(req.Tool.Name) {
                    return &gollem.ToolExecResponse{
                        Error: fmt.Errorf("rate limit exceeded for tool %s", req.Tool.Name),
                    }, nil
                }

                // Execute tool
                resp, err := next(ctx, req)

                // Handle response and errors
                if resp.Error != nil {
                    fmt.Printf("❌ Tool %s failed: %v\n", req.Tool.Name, resp.Error)

                    // Example: Implement fallback mechanisms
                    if req.Tool.Name == "external_api" {
                        // Use cached data as fallback
                        if cachedData := getFromCache(req.Tool.Arguments); cachedData != nil {
                            fmt.Printf("🔄 Using cached data for %s\n", req.Tool.Name)
                            return &gollem.ToolExecResponse{Result: cachedData}, nil
                        }
                    }

                    // For critical tools, abort execution
                    if isCriticalTool(req.Tool.Name) {
                        return resp, err
                    }
                } else {
                    fmt.Printf("✅ Tool %s completed successfully\n", req.Tool.Name)
                    // Log metrics, update dashboards, etc.
                }

                return resp, err
            }
        }),

        // Configure limits
        gollem.WithLoopLimit(30),
    )
}

// Helper functions for the example
func isRateLimited(toolName string) bool {
    // Implement your rate limiting logic
    return false
}

func getFromCache(args map[string]any) map[string]any {
    // Implement cache lookup
    return nil
}

func isCriticalTool(toolName string) bool {
    criticalTools := []string{"database_write", "file_delete", "system_command"}
    for _, critical := range criticalTools {
        if toolName == critical {
            return true
        }
    }
    return false
}
```

## Error Handling and Recovery

Example of robust error handling:

```go
func robustAgentExecution(ctx context.Context, agent *gollem.Agent, prompt string) error {
    // Execute with error handling
    err := agent.Execute(ctx, prompt)
    if err != nil {
        // Handle specific error types
        if errors.Is(err, gollem.ErrLoopLimitExceeded) {
            fmt.Println("⚠️  Conversation took too many steps, but may have completed successfully")
            return nil // Treat as success
        }
        
        if errors.Is(err, context.DeadlineExceeded) {
            fmt.Println("⏰ Operation timed out")
            return fmt.Errorf("operation timed out: %w", err)
        }
        
        // Log error with context
        fmt.Printf("❌ Agent execution failed: %v\n", err)
        return fmt.Errorf("agent execution failed: %w", err)
    }
    
    return nil
}
```

## Streaming Response Example

Example of handling streaming responses:

```go
func streamingExample(ctx context.Context, client gollem.LLMClient) error {
    agent := gollem.New(client,
        gollem.WithResponseMode(gollem.ResponseModeStreaming),
        gollem.WithContentStreamMiddleware(func(next gollem.ContentStreamHandler) gollem.ContentStreamHandler {
            return func(ctx context.Context, req *gollem.ContentRequest) (<-chan *gollem.ContentResponse, error) {
                ch, err := next(ctx, req)
                if err != nil {
                    return nil, err
                }

                outCh := make(chan *gollem.ContentResponse)
                go func() {
                    defer close(outCh)
                    for resp := range ch {
                        // Stream each token to the user interface
                        if len(resp.Texts) > 0 {
                            for _, text := range resp.Texts {
                                fmt.Print(text)
                                // Could also send to WebSocket, SSE, etc.
                                streamToWebSocket(text)
                            }
                        }
                        outCh <- resp
                    }
                }()
                return outCh, nil
            }
        }),
    )

    return agent.Execute(ctx, "Write a detailed explanation of Go's concurrency model")
}

func streamToWebSocket(msg string) error {
    // Implement WebSocket streaming
    return nil
}
```



## Argument Validation

Here's an example of proper argument validation:

```go
func validateArgs(args map[string]any, spec gollem.ToolSpec) error {
    // Check required parameters
    for _, required := range spec.Required {
        if _, exists := args[required]; !exists {
            return fmt.Errorf("required parameter '%s' is missing", required)
        }
    }
    
    // Validate parameter types
    for name, param := range spec.Parameters {
        if value, exists := args[name]; exists {
            if err := validateParameterType(value, param.Type); err != nil {
                return fmt.Errorf("parameter '%s': %w", name, err)
            }
        }
    }
    
    return nil
}

func validateParameterType(value any, expectedType gollem.ParameterType) error {
    switch expectedType {
    case gollem.TypeString:
        if _, ok := value.(string); !ok {
            return fmt.Errorf("expected string, got %T", value)
        }
    case gollem.TypeNumber:
        if _, ok := value.(float64); !ok {
            return fmt.Errorf("expected number, got %T", value)
        }
    case gollem.TypeBoolean:
        if _, ok := value.(bool); !ok {
            return fmt.Errorf("expected boolean, got %T", value)
        }
    }
    return nil
}
```

## Plan Mode with Adaptive Skip

The [plan mode example](../examples/plan_mode/) demonstrates advanced multi-step planning with intelligent skip capabilities:

```go
func planModeExample(ctx context.Context, client gollem.LLMClient) error {
    agent := gollem.New(client,
        gollem.WithTools(&SearchTool{}, &AnalysisTool{}, &ReportTool{}),
    )
    
    // Create plan with balanced execution mode (default settings)
    plan, err := agent.Plan(ctx,
        "Research AI trends, analyze the data, and generate a comprehensive report",
        gollem.WithPlanExecutionMode(gollem.PlanExecutionModeBalanced), // Default mode
        gollem.WithSkipConfidenceThreshold(0.8), // Default threshold
        gollem.WithSkipConfirmationHook(func(ctx context.Context, plan *gollem.Plan, decision gollem.SkipDecision) bool {
            // Auto-approve high confidence decisions
            if decision.Confidence >= 0.9 {
                fmt.Printf("🤖 Auto-approving skip (%.2f): %s\n", 
                    decision.Confidence, decision.SkipReason)
                return true
            }
            
            // Ask user for medium confidence decisions
            if decision.Confidence >= 0.7 {
                fmt.Printf("Skip task? (confidence: %.2f)\n", decision.Confidence)
                fmt.Printf("Reason: %s\n", decision.SkipReason)
                fmt.Printf("Evidence: %s\n", decision.Evidence)
                
                var response string
                fmt.Print("Continue? (y/n): ")
                fmt.Scanln(&response)
                return strings.ToLower(response) == "y"
            }
            
            return false // Deny low confidence decisions
        }),
        gollem.WithPlanToDoUpdatedHook(func(ctx context.Context, plan *gollem.Plan, changes []gollem.PlanToDoChange) error {
            for _, change := range changes {
                if change.Type == gollem.PlanToDoChangeUpdated && 
                   change.NewToDo != nil && change.NewToDo.Status == "Skipped" {
                    fmt.Printf("⏭️  Skipped: %s\n", change.Description)
                }
            }
            return nil
        }),
    )
    if err != nil {
        return fmt.Errorf("failed to create plan: %w", err)
    }
    
    // Execute the plan
    result, err := plan.Execute(ctx)
    if err != nil {
        return fmt.Errorf("failed to execute plan: %w", err)
    }
    
    // Display results and statistics
    fmt.Printf("Plan completed: %s\n", result)
    displayPlanStatistics(plan)
    
    return nil
}

func displayPlanStatistics(plan *gollem.Plan) {
    todos := plan.GetToDos()
    var completed, skipped, failed int
    
    for _, todo := range todos {
        switch todo.Status {
        case "Completed":
            completed++
        case "Skipped":
            skipped++
        case "Failed":
            failed++
        }
    }
    
    fmt.Printf("\n📊 Plan Statistics:\n")
    fmt.Printf("   Total tasks: %d\n", len(todos))
    fmt.Printf("   ✅ Completed: %d\n", completed)
    fmt.Printf("   ⏭️  Skipped: %d\n", skipped)
    fmt.Printf("   ❌ Failed: %d\n", failed)
    fmt.Printf("   📈 Efficiency: %.1f%%\n", 
        float64(completed+skipped)/float64(len(todos))*100)
}
```

### Plan Execution Modes

The plan mode supports three execution modes:

1. **Complete Mode**: Execute all tasks without skipping
   ```go
   gollem.WithPlanExecutionMode(gollem.PlanExecutionModeComplete)
   ```

2. **Balanced Mode**: Smart skipping with confirmation (default)
   ```go
   gollem.WithPlanExecutionMode(gollem.PlanExecutionModeBalanced) // Default mode
   gollem.WithSkipConfidenceThreshold(0.8) // Default threshold
   ```

3. **Efficient Mode**: Aggressive skipping for speed
   ```go
   gollem.WithPlanExecutionMode(gollem.PlanExecutionModeEfficient)
   gollem.WithSkipConfidenceThreshold(0.6) // Lower threshold
   ```

4. **Using Defaults**: Equivalent to Balanced mode with 0.8 threshold
   ```go
   // No options needed - uses defaults
   plan, err := agent.Plan(ctx, "task description")
   ```

### Skip Decision Intelligence

The LLM provides structured skip decisions with:

- **Confidence levels** (0.0-1.0) indicating certainty
- **Detailed reasoning** for why tasks should be skipped
- **Evidence** from previous execution results
- **Transparent decision-making** process

Example skip decision:
```json
{
  "todo_id": "analyze_data",
  "skip_reason": "Data analysis already completed in previous step with comprehensive results",
  "confidence": 0.85,
  "evidence": "Step 2 output contains detailed analysis with 15 key insights identified"
}
```

## Next Steps

- Learn more about [tool creation](tools.md)
- Explore [MCP server integration](mcp.md)
- Check out the [getting started guide](getting-started.md)
- Understand [history management](history.md) for conversation context
- Discover [plan mode capabilities](plan-mode.md) for complex workflows
- Review the [complete documentation](README.md)
