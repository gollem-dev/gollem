# Middleware System

gollem provides a powerful middleware system for monitoring, logging, and controlling agent behavior. Middleware functions wrap the core handlers to add cross-cutting concerns.

## Available Middleware Types

### ContentBlockMiddleware

Wraps synchronous content generation:

```go
gollem.WithContentBlockMiddleware(func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
	return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
		// Pre-processing: log request, validate inputs
		log.Printf("Generating content with %d inputs", len(req.Inputs))

		// Execute core handler
		resp, err := next(ctx, req)

		// Post-processing: log response, track metrics
		if err == nil && len(resp.Texts) > 0 {
			for _, text := range resp.Texts {
				log.Printf("LLM: %s", text)
			}
			metrics.IncrementCounter("llm_messages_total")
		}

		return resp, err
	}
})
```

### ContentStreamMiddleware

Wraps streaming content generation:

```go
gollem.WithContentStreamMiddleware(func(next gollem.ContentStreamHandler) gollem.ContentStreamHandler {
	return func(ctx context.Context, req *gollem.ContentRequest) (<-chan *gollem.ContentResponse, error) {
		ch, err := next(ctx, req)
		if err != nil {
			return nil, err
		}

		// Wrap response channel for processing
		outCh := make(chan *gollem.ContentResponse)
		go func() {
			defer close(outCh)
			for resp := range ch {
				if len(resp.Texts) > 0 {
					for _, text := range resp.Texts {
						fmt.Print(text) // Stream to UI
					}
				}
				outCh <- resp
			}
		}()

		return outCh, nil
	}
})
```

### ToolMiddleware

Wraps tool execution:

```go
gollem.WithToolMiddleware(func(next gollem.ToolHandler) gollem.ToolHandler {
	return func(ctx context.Context, req *gollem.ToolExecRequest) (*gollem.ToolExecResponse, error) {
		// Pre-execution: security checks, logging
		log.Printf("Executing tool: %s", req.Tool.Name)

		// Implement access control
		if !isToolAllowed(req.Tool.Name) {
			return &gollem.ToolExecResponse{
				Error: fmt.Errorf("tool %s not allowed", req.Tool.Name),
			}, nil
		}

		// Execute tool
		resp, err := next(ctx, req)

		// Post-execution: metrics, error handling
		if resp.Error != nil {
			log.Printf("Tool %s failed: %v", req.Tool.Name, resp.Error)
			metrics.IncrementCounter("tool_errors_total", "tool", req.Tool.Name)
		} else {
			log.Printf("Tool %s completed in %dms", req.Tool.Name, resp.Duration)
			metrics.RecordDuration("tool_execution_duration", req.Tool.Name, resp.Duration)
		}

		return resp, err
	}
})
```

## Response Modes

Choose between blocking and streaming responses:

```go
agent := gollem.New(client,
	gollem.WithResponseMode(gollem.ResponseModeStreaming), // Real-time streaming
	gollem.WithContentStreamMiddleware(func(next gollem.ContentStreamHandler) gollem.ContentStreamHandler {
		return func(ctx context.Context, req *gollem.ContentRequest) (<-chan *gollem.ContentResponse, error) {
			ch, err := next(ctx, req)
			if err != nil {
				return nil, err
			}
			// Print tokens as they arrive
			outCh := make(chan *gollem.ContentResponse)
			go func() {
				defer close(outCh)
				for resp := range ch {
					if len(resp.Texts) > 0 {
						for _, text := range resp.Texts {
							fmt.Print(text)
						}
					}
					outCh <- resp
				}
			}()
			return outCh, nil
		}
	}),
)
```

## Practical Examples

### Real-time Streaming to WebSocket

```go
agent := gollem.New(client,
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
					// Broadcast to WebSocket clients
					if len(resp.Texts) > 0 {
						for _, text := range resp.Texts {
							websocketBroadcast(text)
						}
					}
					outCh <- resp
				}
			}()
			return outCh, nil
		}
	}),
	gollem.WithToolMiddleware(func(next gollem.ToolHandler) gollem.ToolHandler {
		return func(ctx context.Context, req *gollem.ToolExecRequest) (*gollem.ToolExecResponse, error) {
			// Notify clients about tool execution
			websocketSend(fmt.Sprintf("Executing: %s", req.Tool.Name))
			return next(ctx, req)
		}
	}),
)
```

### Comprehensive Logging and Monitoring

```go
agent := gollem.New(client,
	gollem.WithContentBlockMiddleware(func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
		return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
			resp, err := next(ctx, req)
			if err == nil {
				logger.Info("LLM response",
					"texts", len(resp.Texts),
					"input_tokens", resp.InputToken,
					"output_tokens", resp.OutputToken)
				metrics.IncrementCounter("llm_messages_total")
			}
			return resp, err
		}
	}),
	gollem.WithToolMiddleware(func(next gollem.ToolHandler) gollem.ToolHandler {
		return func(ctx context.Context, req *gollem.ToolExecRequest) (*gollem.ToolExecResponse, error) {
			logger.Info("Tool execution started",
				"tool", req.Tool.Name,
				"args", req.Tool.Arguments,
				"request_id", ctx.Value("request_id"))

			resp, err := next(ctx, req)

			if resp.Error != nil {
				logger.Error("Tool execution failed",
					"tool", req.Tool.Name,
					"error", resp.Error)
				metrics.IncrementCounter("tool_errors_total", "tool", req.Tool.Name)
			} else {
				logger.Info("Tool execution completed",
					"tool", req.Tool.Name,
					"duration_ms", resp.Duration)
				metrics.RecordDuration("tool_execution_duration", req.Tool.Name, resp.Duration)
			}

			return resp, err
		}
	}),
)
```

### Security and Access Control

```go
agent := gollem.New(client,
	gollem.WithToolMiddleware(func(next gollem.ToolHandler) gollem.ToolHandler {
		return func(ctx context.Context, req *gollem.ToolExecRequest) (*gollem.ToolExecResponse, error) {
			userID := ctx.Value("user_id").(string)

			// Check permissions
			if !hasPermission(userID, req.Tool.Name) {
				return &gollem.ToolExecResponse{
					Error: fmt.Errorf("user %s not authorized for tool %s", userID, req.Tool.Name),
				}, nil
			}

			// Rate limiting
			if isRateLimited(userID, req.Tool.Name) {
				return &gollem.ToolExecResponse{
					Error: fmt.Errorf("rate limit exceeded for user %s", userID),
				}, nil
			}

			return next(ctx, req)
		}
	}),
)
```

### Error Recovery and Fallbacks

```go
agent := gollem.New(client,
	gollem.WithToolMiddleware(func(next gollem.ToolHandler) gollem.ToolHandler {
		return func(ctx context.Context, req *gollem.ToolExecRequest) (*gollem.ToolExecResponse, error) {
			resp, err := next(ctx, req)

			// Implement fallback mechanisms
			if resp.Error != nil {
				switch req.Tool.Name {
				case "external_api":
					// Use cached data as fallback
					if cachedData := getFromCache(req.Tool.Arguments); cachedData != nil {
						log.Printf("Using cached data for %s", req.Tool.Name)
						return &gollem.ToolExecResponse{
							Result: cachedData,
						}, nil
					}
				case "file_operation":
					// Retry with backoff
					if retryCount < maxRetries {
						time.Sleep(backoffDuration(retryCount))
						return next(ctx, req) // Retry
					}
				}
			}

			return resp, err
		}
	}),
	gollem.WithLoopLimit(10), // Prevent infinite loops
)
```

## Built-in Middleware

### Automatic History Compaction (compacter)

The compacter middleware automatically handles token limit errors by compressing conversation history using LLM summarization. When a token limit error is detected, it summarizes the oldest messages (default 70%) and retries the request.

```go
import "github.com/gollem-dev/gollem/middleware/compacter"

agent := gollem.New(client,
	gollem.WithContentBlockMiddleware(
		compacter.NewContentBlockMiddleware(
			client,
			compacter.WithCompactRatio(0.7),     // Compress oldest 70% of history (default)
			compacter.WithMaxRetries(3),         // Max retry attempts (default)
			compacter.WithCompactionHook(func(ctx context.Context, event *compacter.CompactionEvent) {
				// Observability: track compaction events
				log.Printf("Compacted: %d -> %d chars, tokens: in=%d out=%d",
					event.OriginalDataSize,
					event.CompactedDataSize,
					event.InputTokens,
					event.OutputTokens,
				)
			}),
		),
	),
	gollem.WithContentStreamMiddleware(
		compacter.NewContentStreamMiddleware(client),
	),
)
```

**Features:**
- **Automatic Recovery**: Detects `ErrTagTokenExceeded` and automatically compacts history
- **LLM-based Summarization**: Uses the same LLM client to generate high-quality summaries that preserve important context
- **Character-based Compression**: Compacts based on character count (default 70% of oldest messages)
- **Configurable**: Customize compression ratio, max retries, and summary prompt
- **Observability**: Hook for monitoring compaction events with metrics:
  - `OriginalDataSize` / `CompactedDataSize`: Character counts before/after
  - `InputTokens` / `OutputTokens`: Actual LLM token usage for summarization
  - `Summary`: Generated summary text
  - `Attempt`: Retry attempt number

**Example with Custom Settings:**
```go
middleware := compacter.NewContentBlockMiddleware(
	client,
	compacter.WithCompactRatio(0.8),  // Compress 80% of history
	compacter.WithMaxRetries(5),       // Allow up to 5 retries
	compacter.WithSummaryPrompt(`Summarize the conversation concisely, preserving:
- Key decisions and conclusions
- Important facts and context
- Action items and next steps`),
	compacter.WithLogger(logger),      // Custom logger
)
```

## Next Steps

- Learn how to create [custom tools](tools.md)
- Explore [MCP server integration](mcp.md)
- Understand [strategy patterns](strategy.md) for agent behavior
- Review [tracing](tracing.md) for observability
