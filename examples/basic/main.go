package main

import (
	"bufio"
	"context"
	"fmt"
	"os"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/openai"
	"github.com/gollem-dev/gollem/mcp"
)

type MyTool struct{}

func (t *MyTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "my_tool",
		Description: "Returns a greeting",
		Parameters: map[string]*gollem.Parameter{
			"name": {
				Type:        gollem.TypeString,
				Description: "Name of the person to greet",
				Required:    true,
			},
		},
	}
}

func (t *MyTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	name, ok := args["name"].(string)
	if !ok {
		return nil, fmt.Errorf("name is required")
	}
	return map[string]any{"message": fmt.Sprintf("Hello, %s!", name)}, nil
}

func main() {
	ctx := context.Background()

	// Create OpenAI client
	client, err := openai.New(ctx, os.Getenv("OPENAI_API_KEY"))
	if err != nil {
		panic(err)
	}

	// Create MCP client with local server (with custom client info)
	mcpLocal, err := mcp.NewStdio(ctx, "./mcp-server", []string{},
		mcp.WithEnvVars([]string{"MCP_ENV=test"}),
		mcp.WithStdioClientInfo("gollem-basic-example", "1.0.0"))
	if err != nil {
		panic(err)
	}
	defer func() {
		_ = mcpLocal.Close()
	}()

	// Create MCP client with remote server (HTTP transport)
	// StreamableHTTP is now implemented with the official SDK
	mcpRemote, err := mcp.NewStreamableHTTP(ctx, "http://localhost:8080",
		mcp.WithStreamableHTTPClientInfo("gollem-remote-client", "1.0.0"))
	if err != nil {
		fmt.Printf("⚠️  Could not connect to HTTP MCP server: %v\n", err)
		mcpRemote = nil
	}
	if mcpRemote != nil {
		defer func() {
			_ = mcpRemote.Close()
		}()
	}

	// Create MCP client with remote server (SSE transport)
	// SSE is also implemented with the official SDK
	mcpSSE, err := mcp.NewSSE(ctx, "http://localhost:8081",
		mcp.WithSSEClientInfo("gollem-sse-client", "1.0.0"))
	if err != nil {
		fmt.Printf("⚠️  Could not connect to SSE MCP server: %v\n", err)
		mcpSSE = nil
	}
	if mcpSSE != nil {
		defer func() {
			_ = mcpSSE.Close()
		}()
	}

	// Create gollem agent with automatic session management
	var toolSets []gollem.ToolSet
	toolSets = append(toolSets, mcpLocal)
	if mcpRemote != nil {
		toolSets = append(toolSets, mcpRemote)
	}
	if mcpSSE != nil {
		toolSets = append(toolSets, mcpSSE)
	}

	agent := gollem.New(client,
		// Not only MCP servers,
		gollem.WithToolSets(toolSets...),
		// But also you can use your own built-in tools
		gollem.WithTools(&MyTool{}),
		// System prompt for better context
		gollem.WithSystemPrompt("You are a helpful assistant with access to various tools."),
	)

	fmt.Println("🚀 Gollem Agent started! Type 'quit' to exit.")
	fmt.Println("💡 The agent automatically manages conversation history.")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}

		input := scanner.Text()
		if input == "quit" || input == "exit" {
			fmt.Println("👋 Goodbye!")
			break
		}

		// Execute with automatic session management
		// No need to manually handle history - it's managed automatically!
		result, err := agent.Execute(ctx, gollem.Text(input))
		if err != nil {
			fmt.Printf("❌ Error: %v\n", err)
			continue
		}

		// Display conclusion if available
		if result != nil && !result.IsEmpty() {
			fmt.Printf("💭 Conclusion: %s\n", result.String())
		}

		// Optional: Show conversation statistics
		if history, err := agent.Session().History(); err == nil && history != nil {
			fmt.Printf("📊 (Conversation has %d messages)\n", history.ToCount())
		}
	}
}
