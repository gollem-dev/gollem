# Tracing

gollem provides a tracing system for observing agent execution. The `trace` package follows the `slog.Handler` pattern: a `Handler` interface defines the contract, and concrete implementations provide different backends.

## Overview

During agent execution, gollem emits lifecycle events:

- **Agent Execute**: The root span wrapping an `agent.Execute()` call
- **LLM Call**: Each request/response to the LLM provider
- **Tool Exec**: Each tool invocation with arguments and results
- **Sub Agent**: Internal sub-agent invocations within gollem via `SubAgent.Run()`
- **Child Agent**: External agent invocations via `trace.AsChildAgent` (distinguished from internal sub-agents)
- **Event**: Custom events emitted by strategies (e.g., plan creation, reflection)

These events form a tree structure rooted at the agent execution span.

## Handler Interface

All trace backends implement the `trace.Handler` interface:

```go
type Handler interface {
    StartAgentExecute(ctx context.Context) context.Context
    EndAgentExecute(ctx context.Context, err error)

    StartLLMCall(ctx context.Context) context.Context
    EndLLMCall(ctx context.Context, data *LLMCallData, err error)

    StartToolExec(ctx context.Context, toolName string, args map[string]any) context.Context
    EndToolExec(ctx context.Context, result map[string]any, err error)

    StartSubAgent(ctx context.Context, name string) context.Context
    EndSubAgent(ctx context.Context, err error)

    StartChildAgent(ctx context.Context, name string) context.Context
    EndChildAgent(ctx context.Context, err error)

    AddEvent(ctx context.Context, kind string, data any)

    Finish(ctx context.Context) error
}
```

Each `Start*` method returns a new `context.Context` that carries span state. The corresponding `End*` method receives this context to close the span. `Finish` is called after execution completes to perform any final cleanup (e.g., persisting data).

## Enabling Tracing

Pass a `Handler` to the agent via `gollem.WithTrace()`:

```go
agent := gollem.New(client, gollem.WithTrace(handler))
```

## Built-in Handlers

### Recorder (`trace.New()`)

The in-memory recorder collects trace data into a tree of `Span` structs. It is useful for debugging, testing, and persisting traces.

```go
rec := trace.New()
agent := gollem.New(client, gollem.WithTrace(rec))

result, err := agent.Execute(ctx, gollem.Text("Hello"))

// Access trace data
tr := rec.Trace()
fmt.Printf("Trace ID: %s\n", tr.TraceID)
fmt.Printf("Root span: %s (%s)\n", tr.RootSpan.Name, tr.RootSpan.Kind)
for _, child := range tr.RootSpan.Children {
    fmt.Printf("  - %s (%s) duration=%s\n", child.Name, child.Kind, child.Duration)
}
```

#### Custom Trace ID

By default, each trace is assigned a UUID v7. To correlate with external systems (e.g., HTTP request IDs, distributed trace IDs), you can specify a custom trace ID:

```go
rec := trace.New(trace.WithTraceID(requestID))
agent := gollem.New(client, gollem.WithTrace(rec))
```

If `WithTraceID` is not set or set to an empty string, a UUID v7 is generated automatically.

#### Persisting Traces with Repository

Traces can be saved automatically by providing a `Repository`:

```go
rec := trace.New(
    trace.WithRepository(trace.NewFileRepository("./traces")),
    trace.WithMetadata(trace.TraceMetadata{
        Model:    "gpt-4",
        Strategy: "planexec",
        Labels:   map[string]string{"env": "production", "user": "alice"},
    }),
)

agent := gollem.New(client, gollem.WithTrace(rec))
result, err := agent.Execute(ctx, gollem.Text("Analyze the logs"))
// Trace is automatically saved to ./traces/{trace_id}.json on Finish
```

`FileRepository` saves each trace as `{trace_id}.json` in the given directory. A custom trace ID may come from an untrusted source such as a request header, so `Save` rejects any trace ID that cannot be used as a single file name in that directory:

- an empty trace ID
- a trace ID that contains `/` or `\`
- a trace ID that `filepath.IsLocal` rejects. On Windows, this includes IDs that contain `:` and reserved device names such as `NUL`.

`Save` also refuses to write through a symbolic link that is absolute or points outside the directory. In these cases, `Save` returns an error and writes no file. When the recorder is attached with `gollem.WithTrace`, `Agent.Execute` logs this error as a warning and does not return it. If every trace must be saved, use the default UUID, or build custom trace IDs only from letters, digits, `-`, and `_`.

You can implement the `Repository` interface for custom storage (database, cloud storage, etc.):

```go
type Repository interface {
    Save(ctx context.Context, trace *Trace) error
}
```

#### Trace Data Structure

The recorded trace has this structure:

```
Trace
├── TraceID
├── Metadata (model, strategy, labels)
├── StartedAt / EndedAt
└── RootSpan (agent_execute)
    ├── LLM Call span (input/output tokens, model, request/response)
    ├── Tool Exec span (tool name, args, result)
    ├── Sub Agent span (internal sub-agent)
    │   ├── LLM Call span
    │   └── Tool Exec span
    ├── Child Agent span (external agent via AsChildAgent)
    │   ├── LLM Call span
    │   └── Tool Exec span
    └── Event span (strategy-defined events)
```

Each span contains:
- `SpanID`, `ParentID` for the tree structure
- `Kind`: `agent_execute`, `llm_call`, `tool_exec`, `sub_agent`, `event`
- `StartedAt`, `EndedAt`, `Duration` for timing
- `Status`: `ok` or `error`
- Kind-specific data (`LLMCallData`, `ToolExecData`, `EventData`)

#### LLM Call Messages

`LLMCallData.Request.Messages` records only the messages newly added in that turn (e.g. the latest user input and any tool responses), not the full conversation history that was actually sent to the provider. This keeps each span proportional to the work done in that turn — the prior history can be reconstructed by walking the chronologically earlier `llm_call` spans in the same trace.

#### LLM Call Response

`LLMCallData.Response` records what the provider returned in that call:

- `Texts`: the text blocks
- `FunctionCalls`: the tool calls
- `Thoughts`: the same thinking text as `Response.Thoughts` of the call, recorded under the JSON key `thoughts`. For `Stream`, it holds the thinking text of the whole stream: Claude and Gemini record each received fragment as one element, OpenAI Chat Completions and Ollama record the fragments joined into one element, and the OpenAI Responses API records each reasoning summary as one element. A thinking block whose text the provider omitted adds no element. For example, Claude omits the text when the thinking display is `omitted`, which is the default for some models.
- `OtherBlockTypes`: the same values as `Response.OtherBlockTypes` of the call, recorded under the JSON key `other_block_types`: the type of every content block other than text and tool_use, in the order received. Only Claude reports it. A thinking block whose text was omitted is listed as `thinking`, so a span whose response holds only such a block shows `"other_block_types":["thinking"]` instead of no content at all.
- `FinishReason`: the same value as `Response.FinishReason` of the call. For `Stream`, it is the last non-empty finish reason the stream returned. It is recorded even when `Texts` is empty, so a span with no text shows why the generation ended. The value depends on the provider:
  - Claude (Claude API and Vertex AI): `stop_reason`, for example `end_turn`, `max_tokens`, or `refusal`
  - Gemini: the candidate's `finishReason`, for example `STOP`, `MAX_TOKENS`, or `SAFETY`
  - OpenAI Chat Completions: the choice's `finish_reason`, for example `stop`, `length`, or `content_filter`
  - OpenAI Responses API: `incomplete_details.reason` when the response is incomplete (for example `max_output_tokens`), otherwise the response `status` (for example `completed`). The Responses API has no finish reason, so a refusal is reported as `completed`, with the refusal message in `Texts` and in `Refusal`.
  - Ollama: `done_reason`, for example `stop` or `length`

  The JSON key `finish_reason` is omitted when the value is empty.
- `Refusal`: the same values as `Response.Refusal` of the call, recorded under the JSON key `refusal` with `reason`, `categories` and `explanation`. It is omitted when the provider reported no refusal. For Claude it holds `stop_details.category` and `stop_details.explanation`; see [Refusal Details](llm.md#refusal-details) for the other providers. For `Stream`, it is the last refusal the stream returned. When Gemini reports `PROHIBITED_CONTENT`, the call returns an error and the span records the error, and `Response` is still recorded with the finish reason and the refusal, because the span keeps only the error message.

### OpenTelemetry Handler (`trace/otel`)

The `trace/otel` package bridges gollem's trace events to OpenTelemetry spans. This integrates with any OTel-compatible backend such as Jaeger, Zipkin, or OTLP collectors.

```go
import traceOtel "github.com/gollem-dev/gollem/trace/otel"

// Uses the global TracerProvider (set by your OTel SDK setup)
agent := gollem.New(client, gollem.WithTrace(traceOtel.New()))
```

#### Explicit TracerProvider

If you manage multiple `TracerProvider` instances or want to avoid the global:

```go
import (
    traceOtel "github.com/gollem-dev/gollem/trace/otel"
    sdkTrace "go.opentelemetry.io/otel/sdk/trace"
)

tp := sdkTrace.NewTracerProvider(
    sdkTrace.WithBatcher(exporter),
)

agent := gollem.New(client, gollem.WithTrace(
    traceOtel.New(traceOtel.WithTracerProvider(tp)),
))
```

#### Span Mapping

gollem events map to OTel spans as follows:

| gollem Event | OTel Span Name | Span Kind | Attributes |
|---|---|---|---|
| Agent Execute | `agent_execute` | Internal | - |
| LLM Call | `llm_call` | Client | `llm.model`, `llm.input_tokens`, `llm.output_tokens` |
| Tool Exec | `tool:{name}` | Internal | `tool.name`, `tool.args` |
| Sub Agent | `sub_agent:{name}` | Internal | - |
| Child Agent | `child_agent:{name}` | Internal | - |
| Event | _(added as span event)_ | - | `event.data` |

Errors are recorded via `span.RecordError()`. Parent-child relationships are preserved through context propagation.

### Multi Handler (`trace.Multi()`)

`Multi` fans out events to multiple handlers. Each handler receives its own isolated context, so multiple `Recorder` instances or any combination of handlers work without interference.

```go
import (
    "github.com/gollem-dev/gollem/trace"
    traceOtel "github.com/gollem-dev/gollem/trace/otel"
)

// Record in-memory AND export to OTel
rec := trace.New(trace.WithRepository(trace.NewFileRepository("./traces")))
otelHandler := traceOtel.New()

agent := gollem.New(client,
    gollem.WithTrace(trace.Multi(rec, otelHandler)),
)
```

`Finish` collects errors from all handlers using `errors.Join`.

### AsChildAgent Helper (`trace.AsChildAgent()`)

`AsChildAgent` creates a `Handler` that maps a child `Agent.Execute()` into the parent trace tree as a `SpanKindAgentExecute` span. This is useful when running multiple gollem Agents within a single trace.

Without `AsChildAgent`, each `Agent.Execute()` would create its own root span, overwriting the parent trace. With `AsChildAgent`, child agents appear as nested spans in the parent trace tree.

```go
recorder := trace.New(trace.WithRepository(repo))

// Root agent uses the recorder directly
rootAgent := gollem.New(client, gollem.WithTrace(recorder))

// Child agents use AsChildAgent to appear as nested spans
childHandler := trace.AsChildAgent(recorder, "task-1")
childAgent := gollem.New(client,
    gollem.WithTrace(childHandler),
    gollem.WithToolSets(tools...),
)
resp, err := childAgent.Execute(ctx, gollem.Text("Analyze logs"))
```

The resulting trace tree:

```
Root Agent Execute (agent_execute)
├── Child Agent: task-1 (agent_execute)  ← created by AsChildAgent
│   ├── LLM Call
│   └── Tool Exec
├── Child Agent: task-2 (agent_execute)  ← created by AsChildAgent
│   ├── LLM Call
│   └── Sub Agent: searcher (sub_agent)   ← gollem-internal sub-agent
│       └── LLM Call
└── LLM Call (final response)
```

Key distinction:
- **`SpanKindAgentExecute`**: External agents created via `AsChildAgent` — these are full `Agent.Execute()` calls
- **`SpanKindSubAgent`**: Internal sub-agents managed by gollem itself via `SubAgent.Run()`

`AsChildAgent` is thread-safe: multiple child handlers can share the same parent recorder and run concurrently.

`Finish` on the child handler is a no-op — the parent handler owns the `Finish` lifecycle.

## Implementing a Custom Handler

To create your own trace backend, implement the `trace.Handler` interface:

```go
type myHandler struct{}

func (h *myHandler) StartAgentExecute(ctx context.Context) context.Context {
    log.Println("agent execution started")
    return ctx
}

func (h *myHandler) EndAgentExecute(ctx context.Context, err error) {
    if err != nil {
        log.Printf("agent execution failed: %v", err)
    } else {
        log.Println("agent execution completed")
    }
}

func (h *myHandler) StartLLMCall(ctx context.Context) context.Context { return ctx }
func (h *myHandler) EndLLMCall(ctx context.Context, data *trace.LLMCallData, err error) {
    if data != nil {
        log.Printf("LLM call: model=%s tokens=%d/%d", data.Model, data.InputTokens, data.OutputTokens)
    }
}

func (h *myHandler) StartToolExec(ctx context.Context, toolName string, args map[string]any) context.Context {
    return ctx
}
func (h *myHandler) EndToolExec(ctx context.Context, result map[string]any, err error) {}

func (h *myHandler) StartSubAgent(ctx context.Context, name string) context.Context { return ctx }
func (h *myHandler) EndSubAgent(ctx context.Context, err error) {}

func (h *myHandler) StartChildAgent(ctx context.Context, name string) context.Context { return ctx }
func (h *myHandler) EndChildAgent(ctx context.Context, err error) {}

func (h *myHandler) AddEvent(ctx context.Context, kind string, data any) {}

func (h *myHandler) Finish(ctx context.Context) error { return nil }
```

Use `Start*` methods to store span state in the context (via `context.WithValue`) and retrieve it in the corresponding `End*` methods.
