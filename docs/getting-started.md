# Getting Started with gollem

gollem is a Go framework for building applications with Large Language Models (LLMs). This guide will help you get started with the framework.

## Installation

Install gollem using Go modules:

```bash
go get github.com/gollem-dev/gollem
```

## Basic Usage

Here's a simple example of how to use gollem with OpenAI:

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/gollem-dev/gollem"
    "github.com/gollem-dev/gollem/llm/openai"
    "github.com/gollem-dev/gollem/mcp"
)

func main() {
    ctx := context.Background()

    // Create OpenAI client
    client, err := openai.New(ctx, os.Getenv("OPENAI_API_KEY"))
    if err != nil {
        panic(err)
    }

    // Create MCP client (optional)
    mcpClient, err := mcp.NewStreamableHTTP(ctx, "http://localhost:8080")
    if err != nil {
        panic(err)
    }
    defer mcpClient.Close()

    // Create gollem agent with automatic session management
    agent := gollem.New(client,
        gollem.WithToolSets(mcpClient),
        gollem.WithSystemPrompt("You are a helpful assistant."),
        gollem.WithContentBlockMiddleware(func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
            return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
                resp, err := next(ctx, req)
                if err == nil && len(resp.Texts) > 0 {
                    for _, text := range resp.Texts {
                        fmt.Println(text)
                    }
                }
                return resp, err
            }
        }),
    )

    // Execute with automatic session management
    if err := agent.Execute(ctx, "Hello, how are you?"); err != nil {
        panic(err)
    }
}
```

This code uses the OpenAI model to receive a message from the user and send it to the LLM. The `Execute` method automatically manages conversation history, making it easy to build conversational applications.

For information on how to integrate with Tools and MCP servers, please refer to [tools](tools.md) and [mcp](mcp.md) documents.

## Supported LLM Providers

gollem supports multiple LLM providers:

- **Gemini** - Google's Gemini models
- **Anthropic (Claude)** - Anthropic's Claude models  
- **OpenAI** - OpenAI's GPT models

Each provider has its own client implementation in the `llm` package. See the respective documentation for configuration options.

## Key Concepts

1. **LLM Client**: The interface to communicate with LLM providers
2. **Agent**: The main interface that manages sessions and provides the `Execute` method
3. **Tools**: Custom functions that LLMs can use to perform actions (see [Tools](tools.md))
4. **MCP Server**: External tool integration through Model Context Protocol (see [MCP Server Integration](mcp.md))
5. **Automatic Session Management**: Built-in conversation history management
6. **Hooks**: Callback functions for monitoring and controlling agent behavior

## Agent vs Direct Session Usage

gollem provides two main approaches:

### Agent Approach (Recommended)
Use the `Agent` with `Execute` method for conversational applications:

```go
agent := gollem.New(client,
    gollem.WithTools(tools...),
    gollem.WithSystemPrompt("You are helpful."),
)

// Automatic history management
err := agent.Execute(ctx, "Hello")
err = agent.Execute(ctx, "Continue our conversation") // Remembers previous context
```

### Direct Session Approach
Use direct session for simple, one-off queries:

```go
session, err := client.NewSession(ctx)
if err != nil {
    panic(err)
}

result, err := session.Generate(ctx, []gollem.Input{gollem.Text("Hello")})
if err != nil {
    panic(err)
}
fmt.Println(result.Texts)
```

## Error Handling

gollem provides robust error handling capabilities to help you build reliable applications:

### Error Types
- **LLM Errors**: Errors from the LLM provider (e.g., rate limits, invalid requests)
- **Tool Execution Errors**: Errors during tool execution
- **MCP Server Errors**: Errors from MCP server communication



Example of error handling:
```go
if err := agent.Execute(ctx, userInput); err != nil {
    // Handle errors appropriately
    log.Printf("Error: %v", err)
    return fmt.Errorf("failed to process request: %w", err)
}
```

## Session Management

### Automatic Session Management (Recommended)
The `Execute` method automatically manages conversation history:

```go
agent := gollem.New(client, gollem.WithTools(tools...))

// First interaction
err := agent.Execute(ctx, "Hello, I'm working on a project.")

// Follow-up (automatically remembers previous context)
err = agent.Execute(ctx, "Can you help me with the next step?")

// Access conversation history if needed
history := agent.Session().History()
```

### Manual History Management (Legacy)
For backward compatibility, manual history management is still supported:

```go
// Legacy approach (not recommended)
var history *gollem.History
newHistory, err := agent.Prompt(ctx, "Hello", gollem.WithHistory(history))
if err != nil {
    return err
}
history = newHistory
```

## Middleware System

gollem provides comprehensive middleware for monitoring and controlling agent behavior:

```go
agent := gollem.New(client,
    gollem.WithContentBlockMiddleware(func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
        return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
            // Process each message from the LLM
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
            // Monitor tool execution
            fmt.Printf("⚡ Executing: %s\n", req.Tool.Name)

            resp, err := next(ctx, req)

            // Handle tool errors
            if resp.Error != nil {
                fmt.Printf("❌ Tool %s failed: %v\n", req.Tool.Name, resp.Error)
            }

            return resp, err
        }
    }),
)
```

## Next Steps

- Learn how to create and use [custom tools](tools.md)
- Explore [MCP server integration](mcp.md)
- Understand [middleware](middleware.md) for monitoring and control
- Explore [strategy patterns](strategy.md) for agent behavior
- Understand [history management](history.md) for conversation context
- Check out [practical examples](examples.md)
- Review the [complete documentation](README.md)
