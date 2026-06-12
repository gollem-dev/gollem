# 🤖 gollem [![Go Reference](https://pkg.go.dev/badge/github.com/gollem-dev/gollem.svg)](https://pkg.go.dev/github.com/gollem-dev/gollem) [![Test](https://github.com/gollem-dev/gollem/actions/workflows/test.yml/badge.svg)](https://github.com/gollem-dev/gollem/actions/workflows/test.yml) [![Lint](https://github.com/gollem-dev/gollem/actions/workflows/lint.yml/badge.svg)](https://github.com/gollem-dev/gollem/actions/workflows/lint.yml) [![Gosec](https://github.com/gollem-dev/gollem/actions/workflows/gosec.yml/badge.svg)](https://github.com/gollem-dev/gollem/actions/workflows/gosec.yml) [![Trivy](https://github.com/gollem-dev/gollem/actions/workflows/trivy.yml/badge.svg)](https://github.com/gollem-dev/gollem/actions/workflows/trivy.yml)

GO for Large LanguagE Model (GOLLEM)

<p align="center">
  <img src="./docs/images/logo.png" height="128" />
</p>

`gollem` provides:
- **Common interface** to query prompt to Large Language Model (LLM) services
  - Generate / Stream: Generate text content from prompt (with per-call option overrides)
  - GenerateEmbedding: Generate embedding vector from text (OpenAI and Gemini)
- **Framework for building agentic applications** of LLMs with
  - Tools by MCP (Model Context Protocol) server and your built-in tools
  - Automatic session management for continuous conversations
  - Portable conversational memory with history for stateless/distributed applications
  - Intelligent memory management with automatic history compaction
  - Middleware system for monitoring, logging, and controlling agent behavior

## Supported LLMs

- [x] **Gemini** (see [models](https://ai.google.dev/gemini-api/docs/models?hl=ja))
- [x] **Anthropic Claude** (see [models](https://docs.anthropic.com/en/docs/about-claude/models/all-models))
  - Direct access via Anthropic API
  - Via Google Vertex AI (see [LLM Provider Configuration](docs/llm.md#claude-vertex-ai))
- [x] **OpenAI** (see [models](https://platform.openai.com/docs/models))

## Install

```bash
go get github.com/gollem-dev/gollem
```

## Quick Start

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

	// Create LLM client
	client, err := openai.New(ctx, os.Getenv("OPENAI_API_KEY"))
	if err != nil {
		panic(err)
	}

	// Create session for one-time query
	session, err := client.NewSession(ctx)
	if err != nil {
		panic(err)
	}

	// Generate content
	result, err := session.Generate(ctx, []gollem.Input{gollem.Text("Hello, how are you?")})
	if err != nil {
		panic(err)
	}

	fmt.Println(result.Texts)
}
```

## Features

### Agent Framework

Build conversational agents with automatic session management and tool integration. [Learn more →](docs/getting-started.md)

```go
agent := gollem.New(client,
	gollem.WithTools(&GreetingTool{}),
	gollem.WithSystemPrompt("You are a helpful assistant."),
)

// Session is managed automatically across calls
agent.Execute(ctx, "Hello!")
agent.Execute(ctx, "What did I just say?") // remembers context
```

### Tool Integration

Define custom tools for LLMs to call, or connect external tools via MCP. [Tools →](docs/tools.md) | [MCP →](docs/mcp.md)

```go
// Custom tool - implement Spec() and Run()
type SearchTool struct{}

func (t *SearchTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "search",
		Description: "Search the database",
		Parameters:  map[string]*gollem.Parameter{
			"query": {Type: gollem.TypeString, Description: "Search query"},
		},
	}
}

func (t *SearchTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return map[string]any{"results": doSearch(args["query"].(string))}, nil
}

// MCP server - connect external tool servers
mcpClient, _ := mcp.NewStdio(ctx, "./mcp-server", []string{})
agent := gollem.New(client,
	gollem.WithTools(&SearchTool{}),
	gollem.WithToolSets(mcpClient),
)
```

### Multimodal Input

Send images and PDFs alongside text prompts. [Learn more →](docs/llm.md#pdf-input-support)

```go
img, _ := gollem.NewImage(imageBytes)
pdf, _ := gollem.NewPDFFromReader(file)

result, _ := session.Generate(ctx, []gollem.Input{img, pdf, gollem.Text("Describe these.")})
```

### Structured Output

Constrain LLM responses to a JSON Schema. [Learn more →](docs/schema.md)

```go
schema, _ := gollem.ToSchema(UserProfile{})
session, _ := client.NewSession(ctx,
	gollem.WithSessionContentType(gollem.ContentTypeJSON),
	gollem.WithSessionResponseSchema(schema),
)
resp, _ := session.Generate(ctx, []gollem.Input{gollem.Text("Extract: John, 30, john@example.com")})
// resp.Texts[0] is valid JSON matching the schema
```

For one-shot queries, `Query[T]()` combines schema generation, session creation, LLM call, and JSON parsing into a single generic function call with automatic retry on parse failures:

```go
type UserProfile struct {
	Name  string `json:"name" description:"User's full name"`
	Age   int    `json:"age" description:"Age in years"`
	Email string `json:"email" description:"Email address"`
}

result, _ := gollem.Query[UserProfile](ctx, client, "Extract: John, 30, john@example.com",
	gollem.WithQuerySystemPrompt("You are a data extractor."),
)
// result.Data is *UserProfile — type-safe, already parsed
```

To run a structured query on an **existing session** (preserving conversation history), use `SessionQuery[T]()`:

```go
// session already has conversation context from prior Generate calls
resp, _ := gollem.SessionQuery[UserProfile](ctx, session, "Who am I?")
// resp.Data is *UserProfile, parsed from the LLM's JSON response
// The session's history (including this exchange) is preserved
```

### Middleware

Monitor, log, and control agent behavior with composable middleware. [Learn more →](docs/middleware.md)

```go
agent := gollem.New(client,
	gollem.WithToolMiddleware(func(next gollem.ToolHandler) gollem.ToolHandler {
		return func(ctx context.Context, req *gollem.ToolExecRequest) (*gollem.ToolExecResponse, error) {
			log.Printf("Tool called: %s", req.Tool.Name)
			return next(ctx, req)
		}
	}),
)
```

### Strategy Pattern

Swap execution strategies: simple, ReAct, or Plan & Execute. [Learn more →](docs/strategy.md)

```go
import "github.com/gollem-dev/gollem/strategy/planexec"

agent := gollem.New(client,
	gollem.WithStrategy(planexec.New(client)),
	gollem.WithTools(&SearchTool{}, &AnalysisTool{}),
)
```

### Tracing

Observe agent execution with pluggable backends (in-memory, OpenTelemetry). [Learn more →](docs/tracing.md)

```go
import "github.com/gollem-dev/gollem/trace"

rec := trace.New(trace.WithRepository(trace.NewFileRepository("./traces")))
agent := gollem.New(client, gollem.WithTrace(rec))
```

<p align="center">
  <img width="860" src="https://github.com/user-attachments/assets/6b9d77e0-d580-4c08-b7c8-3b2b6cd733eb" />
</p>

### History Management

Portable conversation history for stateless/distributed applications. [Learn more →](docs/history.md)

```go
// Export history for persistence
history := agent.Session().History()
data, _ := json.Marshal(history)

// Restore in another process
var restored gollem.History
json.Unmarshal(data, &restored)
agent := gollem.New(client, gollem.WithHistory(&restored))
```

For automatic persistence, implement `HistoryRepository` and pass it via `WithHistoryRepository`. gollem then loads history at the start of a session and saves it after every LLM round-trip — no manual marshaling required.

```go
agent := gollem.New(client,
    gollem.WithHistoryRepository(repo, "session-id"),
)

// History is loaded automatically on first Execute, and saved after each round-trip
err := agent.Execute(ctx, gollem.Text("Hello!"))
```

## Examples

See the [examples](https://github.com/gollem-dev/gollem/tree/main/examples) directory for complete working examples:

- **[Simple](examples/simple)**: Minimal example for getting started
- **[Query](examples/query)**: Type-safe structured query with `Query[T]()`
- **[Basic](examples/basic)**: Simple agent with custom tools
- **[Chat](examples/chat)**: Interactive chat application
- **[MCP](examples/mcp)**: Integration with MCP servers
- **[Tools](examples/tools)**: Custom tool development
- **[JSON Schema](examples/json_schema)**: Structured output with JSON Schema validation
- **[Embedding](examples/embedding)**: Text embedding generation
- **[Tracing](examples/tracing)**: Agent execution tracing with file persistence

## Documentation

- **[Getting Started Guide](docs/getting-started.md)**
- **[Tool Development](docs/tools.md)**
- **[MCP Integration](docs/mcp.md)**
- **[Structured Output with JSON Schema](docs/schema.md)**
- **[Middleware System](docs/middleware.md)**
- **[Strategy Pattern](docs/strategy.md)**
- **[Tracing](docs/tracing.md)**
- **[History Management](docs/history.md)**
- **[LLM Provider Configuration](docs/llm.md)**
- **[Debugging](docs/debugging.md)**
- **[API Reference](https://pkg.go.dev/github.com/gollem-dev/gollem)**

## License

Apache 2.0 License. See [LICENSE](LICENSE) for details.
