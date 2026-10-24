# LLM Provider Configuration

This document provides detailed configuration options for each LLM provider supported by gollem.

## Table of Contents

- [Gemini](#gemini)
- [Claude (Anthropic)](#claude-anthropic)
- [Claude (Vertex AI)](#claude-vertex-ai)
- [OpenAI](#openai)
- [Ollama](#ollama)

## Gemini

### Basic Setup

```go
import (
    "context"
    "github.com/gollem-dev/gollem/llm/gemini"
)

client, err := gemini.New(ctx, "your-project-id", "us-central1")
```

### Authentication

Gemini uses Google Cloud credentials. Set up authentication using one of:

```bash
# Option 1: Service account key
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"

# Option 2: gcloud CLI
gcloud auth application-default login

# Option 3: Workload identity (automatic in GKE/Cloud Run)
```

### Configuration Options

#### Model Selection

```go
client, err := gemini.New(ctx, projectID, location,
    gemini.WithModel("gemini-1.5-pro-latest"),
)
```

Available models (default: `gemini-3.8-flash`):
- `gemini-3.8-flash` - Latest Flash model for long-horizon agentic and coding tasks (uses thinking levels; does not accept `MINIMAL`)
- `gemini-3.5-flash` - Flash model with improved agent execution and coding (uses thinking levels)
- `gemini-2.5-pro` - Advanced model with state-of-the-art thinking capabilities
- `gemini-2.5-flash` - Strong price-performance model with well-rounded capabilities
- `gemini-2.5-flash-lite` - Optimized for cost efficiency and low latency
- `gemini-2.0-flash` - Superior speed with native tool use and 1M token context
- `gemini-2.0-flash-thinking-exp-1219` - Experimental model with thinking capabilities

Note: Gemini 1.5 models are deprecated as of April 2025 for new projects.

#### Thinking Level (Gemini 3.x)

Gemini 3.x replaces the numeric `thinking_budget` with a discrete `thinking_level`:

```go
// Use minimal thinking for fast, simple responses
client, err := gemini.New(ctx, projectID, location,
    gemini.WithModel("gemini-3.5-flash"),
    gemini.WithThinkingLevel(gemini.ThinkingLevelMinimal),
)

// Use higher levels for harder reasoning tasks
client, err := gemini.New(ctx, projectID, location,
    gemini.WithModel("gemini-3.8-flash"),
    gemini.WithThinkingLevel(gemini.ThinkingLevelHigh),
)
```

Available levels (lowest → highest): `gemini.ThinkingLevelMinimal`, `gemini.ThinkingLevelLow`, `gemini.ThinkingLevelMedium`, `gemini.ThinkingLevelHigh`. Each value is sent to the API unchanged, so a level that the API adds later can be passed as `gemini.ThinkingLevel("NEW_LEVEL")`.

Without `WithThinkingLevel` (or `WithThinkingBudget`), gollem sends no thinking configuration at all and each model applies its own default — `MEDIUM` for Gemini 3.5 / 3.6 / 3.7 / 3.8 Flash, `HIGH` for Gemini 3 Pro, `MINIMAL` for Gemini 3.5 Flash Lite. Not every model accepts every level (`gemini-3.7-flash` and `gemini-3.8-flash` reject `MINIMAL` with HTTP 400, and `gemini-3-pro-preview` takes only `LOW` and `HIGH`), so set the level only when you know the model supports it.

Note: Gemini 3.x also deprecates `temperature`, `top_p`, and `top_k` — omit those options when using 3.x models.

#### Thought Summaries

Gemini returns the model's reasoning text only when thought summaries are requested. Without this option `Response.Thoughts` is always empty:

```go
client, err := gemini.New(ctx, projectID, location,
    gemini.WithIncludeThoughts(true),
)
```

This affects the reasoning text only. The thought signatures that Gemini 3.x requires for multi-turn tool calling are returned and stored in history regardless of this setting, in both blocking and streaming modes.

#### Thinking Tokens in `OutputToken`

Gemini bills thinking tokens at the output token price (see [Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking)). `Response.OutputToken` therefore reports `candidatesTokenCount + thoughtsTokenCount` from the response's usage metadata, which matches the billed output and the `OutputToken` of the other providers. Calls with thinking enabled report a larger `OutputToken` than the visible response text alone.

#### Thinking Budget (Gemini 2.x)

For Gemini 2.x models, control thinking via a numeric token budget:

```go
// Automatic thinking budget (model decides based on complexity)
client, err := gemini.New(ctx, projectID, location,
    gemini.WithThinkingBudget(-1),
)

// Fixed token budget for thinking
client, err := gemini.New(ctx, projectID, location,
    gemini.WithThinkingBudget(1000), // 1000 tokens
)

// Disable thinking
client, err := gemini.New(ctx, projectID, location,
    gemini.WithThinkingBudget(0),
)
```

The thinking budget controls computational effort for internal reasoning:
- **-1**: Automatic mode - the model decides based on task complexity
- **Positive value**: Fixed token budget for thinking
- **0**: Disable thinking mode

This feature is particularly useful for complex reasoning tasks where you want the model to spend more time thinking through problems before responding.

#### Temperature and Other Parameters

```go
client, err := gemini.New(ctx, projectID, location,
    gemini.WithTemperature(0.7),
    gemini.WithMaxTokens(8192),  // Optional, omit for model's max capacity
    gemini.WithTopP(0.9),
)
```

#### Issuer Scope

```go
client, err := gemini.New(ctx, projectID, location,
    gemini.WithIssuerScope("project-a"),
)
```

Thought signatures and thinking in a history are sent back only to a client with the same model and the same scope. The scope is empty by default. Set different scopes when clients of the same model must not exchange that data. See [Provider-bound data](history.md#provider-bound-data).

### Environment Variables

- `GEMINI_PROJECT_ID` - Google Cloud project ID
- `GEMINI_LOCATION` - Vertex AI location (e.g., "us-central1")
- `GOLLEM_LOGGING_GEMINI_PROMPT` - Enable prompt logging for debugging
- `GOLLEM_LOGGING_GEMINI_RESPONSE` - Enable response logging for debugging

## Claude (Anthropic)

### Basic Setup

```go
import (
    "context"
    "github.com/gollem-dev/gollem/llm/claude"
)

client, err := claude.New(ctx, "your-api-key")
```

### Configuration Options

#### Model Selection

```go
client, err := claude.New(ctx, apiKey,
    claude.WithModel("claude-sonnet-4-5-20250929"),
)
```

Available models:
- `claude-sonnet-4-5-20250929` - Latest Sonnet 4.5 model (default)
- `claude-opus-4-1-20250805` - Most powerful model, best for complex tasks (August 2025)
- `claude-sonnet-4-20250514` - Balanced performance and efficiency
- `claude-3-5-sonnet-20241022` - Previous generation, still widely available
- `claude-3-5-haiku-20241022` - Fast, cost-effective model

Note: Claude Opus 4.1 and Sonnet 4 are hybrid models offering both instant and extended thinking modes.

#### Temperature, Top-P and Max Tokens

```go
client, err := claude.New(ctx, apiKey,
    claude.WithTemperature(0.7),  // Optional: use either temperature OR top_p, not both
    // claude.WithTopP(0.9),      // Alternative to temperature
    claude.WithMaxTokens(8192),   // Optional, see "Max tokens" below
)
```

**Note**: Claude Sonnet 4.5 does not allow both `temperature` and `top_p` to be specified simultaneously. Use one or the other.

**Note**: Recent Claude models return HTTP 400 when `temperature` or `top_p` is set to a non-default value. On those models, control the depth of the response with `WithEffort` instead.

#### Effort

`WithEffort` sends `output_config.effort` on every request of the client's sessions. The effort level controls how many tokens Claude spends on a response, including text, tool calls, and thinking.

```go
client, err := claude.New(ctx, apiKey,
    claude.WithModel("claude-opus-5-5"),
    claude.WithEffort(claude.EffortHigh),
)
```

| Constant | Value sent |
| --- | --- |
| `claude.EffortLow` | `low` |
| `claude.EffortMedium` | `medium` |
| `claude.EffortHigh` | `high` |
| `claude.EffortXHigh` | `xhigh` |
| `claude.EffortMax` | `max` |

When `WithEffort` is not called, gollem does not send `effort`, and the model default applies. The default model, `claude-sonnet-4-5-20250929`, does not support effort, so select a model that supports it with `WithModel`. Which levels a model accepts depends on the model; see the [Anthropic effort documentation](https://platform.claude.com/docs/en/build-with-claude/effort). gollem does not check the level against the model, so the API rejects a level the model does not support. For Vertex AI, use `claude.WithVertexEffort` with the same constants.

#### Thinking configuration

`WithAdaptiveThinking` sends `thinking` with the type `adaptive` on every `Generate` and `Stream` request of the client's sessions. Its argument is sent as `thinking.display`, which decides whether the thinking text is returned.

```go
client, err := claude.New(ctx, apiKey,
    claude.WithModel("claude-sonnet-5-5"),
    claude.WithAdaptiveThinking(claude.ThinkingDisplaySummarized),
)
```

| Argument | `thinking` sent | Thinking text |
| --- | --- | --- |
| `claude.ThinkingDisplaySummarized` | `{"type":"adaptive","display":"summarized"}` | A summary is returned in `Response.Thoughts` |
| `claude.ThinkingDisplayOmitted` | `{"type":"adaptive","display":"omitted"}` | Thinking blocks are returned with empty text |
| `""` | `{"type":"adaptive"}` | The model's default display applies |

When `WithAdaptiveThinking` is not called, gollem sends no `thinking`, and the model's defaults decide whether the model thinks and whether the thinking text is returned. According to the [Anthropic thinking documentation](https://platform.claude.com/docs/en/build-with-claude/thinking), Claude Opus 4.6 does not think when `thinking` is not sent. Claude Sonnet 5.5 thinks adaptively and returns its thinking blocks with empty text, because its default display is `omitted`. To receive the thinking text of such a model, pass `claude.ThinkingDisplaySummarized`.

gollem does not check the setting against the model. A model that does not accept adaptive thinking rejects the request with HTTP 400: `Generate` returns the error, and `Stream` returns it in `Response.Error`. Claude Opus 4.5, Sonnet 4.5 and Haiku 4.5 reject it, so the default model `claude-sonnet-4-5-20250929` does too. See the Anthropic documentation for the settings each model accepts. To control how much the model thinks, use [`WithEffort`](#effort). For Vertex AI, use `claude.WithVertexAdaptiveThinking` with the same constants.

#### Max tokens

The Anthropic Messages API requires `max_tokens` on every request, so gollem always sends a value. When `WithMaxTokens` is not called, gollem sends the model's documented maximum output tokens:

| Model | Max output tokens |
| --- | --- |
| Fable 5, Mythos 5, Opus 5, Opus 4.8, Opus 4.7, Opus 4.6, Sonnet 5, Sonnet 4.6 | 128000 |
| Opus 4.5, Sonnet 4.5, Haiku 4.5 | 64000 |
| Any other model | 64000 |

Model IDs are matched in all three forms: the Claude API dated form (`claude-sonnet-4-5-20250929`), the alias form (`claude-sonnet-4-5`), and the Vertex AI form (`claude-sonnet-4-5@20250929`). Only an 8 digit date suffix is stripped before matching, so an ID such as `claude-opus-5@custom` stays distinct from `claude-opus-5` and falls back rather than inheriting its limit.

A model that is not in the table — a model released after this table was written, or a model served through a compatible endpoint configured with `WithBaseURL` — falls back to 64000. If that model's real limit is lower, call `WithMaxTokens` explicitly.

`WithMaxTokens` always wins over the resolved value, and a per-call `gollem.WithMaxTokens` wins over both. An explicit value is sent as given and never adjusted, so a value the API does not accept — zero, a negative number, or one above the model's limit — is rejected by the API rather than corrected by gollem.

**Note**: gollem previously defaulted to 8192 regardless of model. If your code relied on that cap to bound output length or cost, call `WithMaxTokens(8192)` explicitly.

**Note**: `max_tokens` is a ceiling, not a reservation — it does not by itself make a request slower or more expensive. But a response that actually approaches the ceiling takes time to generate, and `Generate` waits for the whole response. The default request timeout is 30 seconds (`WithTimeout`); raise it before expecting long outputs. The timeout also covers a streamed response from the request to its last event, so a `Stream` call that runs longer is cut off as well.

#### Structured Outputs

By default, a response schema is sent as structured outputs (`output_config.format`) to models that support it, and the API constrains decoding to the schema. If you prefer the schema stated in the conversation over constrained decoding, disable structured outputs so that the schema is sent in a user message instead:

```go
client, err := claude.New(ctx, apiKey,
    claude.WithStructuredOutputsDisabled(),
)
```

With structured outputs disabled, no `output_config.format` is sent, the system prompt and tool list stay the same with or without a schema, and the schema message is kept in the history. `output_config.effort` is still sent when `WithEffort` is set. For Vertex AI, use `claude.WithVertexStructuredOutputsDisabled`, which behaves the same way. See [Provider-Specific Behavior](schema.md#claude).

#### Thinking and Issuer Scope

The text of each thinking block is returned in `Response.Thoughts`. A block whose text the API omitted, and a redacted thinking block, add no element; their types are still listed in `Response.OtherBlockTypes` (see [Finish Reason](#finish-reason)). `Stream` uses the streaming Messages API and returns thinking and text deltas and tool calls as they arrive. Thinking blocks and their signatures are kept in the history in both `Generate` and `Stream`.

```go
client, err := claude.New(ctx, apiKey,
    claude.WithIssuerScope("tenant-a"),
)
```

Thinking blocks in a history are sent back only to a client with the same model and the same scope. The scope is empty by default. Set a different scope for each Anthropic account, because the API rejects a signature created with another account. See [Provider-bound data](history.md#provider-bound-data).

### Environment Variables

- `ANTHROPIC_API_KEY` - Anthropic API key
- `GOLLEM_LOGGING_CLAUDE_PROMPT` - Enable prompt logging
- `GOLLEM_LOGGING_CLAUDE_RESPONSE` - Enable response logging

## Claude (Vertex AI)

### Basic Setup

```go
import (
    "context"
    "github.com/gollem-dev/gollem/llm/claude"
)

client, err := claude.NewWithVertex(ctx, "us-central1", "your-project-id")
```

### Configuration Options

#### Model Selection

```go
client, err := claude.NewWithVertex(ctx, region, projectID,
    claude.WithVertexModel("claude-sonnet-4@20250514"),
)
```

Available models on Vertex AI:
- `claude-opus-4-1@20250805` - Most powerful model (if available in your region)
- `claude-sonnet-4@20250514` - Latest Claude Sonnet model
- `claude-3-5-sonnet@20241022` - Previous generation Sonnet
- `claude-3-5-haiku@20241022` - Fast, cost-effective model

#### Max Tokens

```go
client, err := claude.NewWithVertex(ctx, region, projectID,
    claude.WithVertexMaxTokens(8192),  // Optional
)
```

`WithVertexMaxTokens` follows the same rules as `claude.WithMaxTokens` — see [Max tokens](#max-tokens) above. Vertex AI model IDs use `@` as the version separator (`claude-sonnet-4-5@20250929`) and are matched against the same table.

#### Effort

```go
client, err := claude.NewWithVertex(ctx, region, projectID,
    claude.WithVertexModel(modelID), // a Vertex AI model ID of a model that supports effort
    claude.WithVertexEffort(claude.EffortMedium),
)
```

`WithVertexEffort` follows the same rules as `claude.WithEffort` — see [Effort](#effort) above. The default Vertex AI model, `claude-sonnet-4@20250514`, does not support effort, so select a model that supports it with `WithVertexModel`.

#### Thinking configuration

```go
client, err := claude.NewWithVertex(ctx, region, projectID,
    claude.WithVertexModel(modelID), // a Vertex AI model ID of a model that accepts adaptive thinking
    claude.WithVertexAdaptiveThinking(claude.ThinkingDisplaySummarized),
)
```

`WithVertexAdaptiveThinking` follows the same rules as `claude.WithAdaptiveThinking` — see [Thinking configuration](#thinking-configuration) above. When it is not called, gollem sends no `thinking`, and the model's defaults apply.

#### System Prompt

```go
client, err := claude.NewWithVertex(ctx, region, projectID,
    claude.WithVertexSystemPrompt("You are a helpful assistant."),
)
```

#### Structured Outputs

By default, a response schema is sent as structured outputs (`output_config.format`) to models that support it. If your Google Cloud organization policy does not allow the `structured_outputs` feature for the model, Vertex AI rejects those calls with a 400 naming `constraints/vertexai.allowedPartnerModelFeatures`. Either have an administrator add `publishers/anthropic/models/<model>:structured_outputs` to the policy's allowed values, or disable structured outputs so that the schema is sent in a user message instead:

```go
client, err := claude.NewWithVertex(ctx, region, projectID,
    claude.WithVertexStructuredOutputsDisabled(),
)
```

`WithVertexStructuredOutputsDisabled` behaves the same as `claude.WithStructuredOutputsDisabled` — see [Structured Outputs](#structured-outputs) above.

#### Issuer Scope

```go
client, err := claude.NewWithVertex(ctx, region, projectID,
    claude.WithVertexIssuerScope("vertex:"+projectID),
)
```

`WithVertexIssuerScope` works like `claude.WithIssuerScope` (see [Thinking and Issuer Scope](#thinking-and-issuer-scope)). A Vertex AI client and an Anthropic API client with the same model and scope send each other's thinking blocks. Whether the two services accept each other's signatures has not been confirmed, so set different scopes when you use both.

### Authentication

Uses Google Cloud credentials (same as Gemini):

```bash
# Option 1: Service account key
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"

# Option 2: gcloud CLI
gcloud auth application-default login
```

### Benefits of Vertex AI Integration

- Unified Google Cloud billing and cost management
- Enterprise security with VPC, private endpoints, and audit logs
- Regional deployment for data residency requirements
- Vertex AI MLOps integration for monitoring and management

## OpenAI

### Basic Setup

```go
import (
    "context"
    "github.com/gollem-dev/gollem/llm/openai"
)

client, err := openai.New(ctx, "your-api-key")
```

### Configuration Options

#### Model Selection

```go
client, err := openai.New(ctx, apiKey,
    openai.WithModel("gpt-4-turbo-preview"),
)
```

Available models:
- `o3-pro` - Most powerful reasoning model with extended thinking
- `o3` - Advanced reasoning model for complex tasks
- `o4-mini` - Fast, cost-efficient reasoning model
- `gpt-4.1` - Latest GPT model with 1M token context (June 2024 cutoff)
- `gpt-4.1-mini` - Smaller version of GPT-4.1
- `gpt-4o` - Previous generation, still available
- `gpt-4o-mini` - Smaller, faster GPT-4o variant
- `gpt-3.5-turbo` - Legacy model, cost-effective

Note: GPT-4.5 is in research preview. o1 models are being phased out in favor of o3/o4 series.

#### Temperature and Other Parameters

```go
client, err := openai.New(ctx, apiKey,
    openai.WithTemperature(0.7),
    openai.WithMaxTokens(4096),  // Optional, omit for infinity (model's max)
    openai.WithTopP(0.9),
    openai.WithFrequencyPenalty(0.5),
    openai.WithPresencePenalty(0.5),
)
```

#### Reasoning Effort and Verbosity

```go
client, err := openai.New(ctx, apiKey,
    openai.WithModel("gpt-5.6"),
    openai.WithReasoningEffort("low"),
    openai.WithVerbosity("low"),
)
```

Both values are sent unchanged: the effort as `reasoning_effort` on Chat Completions and `reasoning.effort` on the Responses API, the verbosity as `verbosity` on Chat Completions and `text.verbosity` on the Responses API. Without these options gollem sends neither parameter and each model applies its own default. Not every model accepts every value (`gpt-5.6` rejects `reasoning_effort: "minimal"`, and `gpt-4.1` accepts only `verbosity: "medium"`); the API rejects an unsupported value with HTTP 400, so set a value only when you know the model supports it.

Earlier versions of gollem sent `reasoning_effort: "minimal"` and `verbosity: "low"` when these options were not given. The model defaults can use more output tokens and take longer than those values. To keep the earlier behavior with `gpt-5`, set `openai.WithReasoningEffort("minimal")` and `openai.WithVerbosity("low")` explicitly.

#### System Prompt

```go
client, err := openai.New(ctx, apiKey,
    openai.WithSystemPrompt("You are concise."), // used when the session sets no system prompt
)
```

The system prompt is sent on every request and is not stored in the session history: as a `system` message at the head of the messages on Chat Completions, and as `instructions` on the Responses API. A content middleware can replace it through `ContentRequest.SystemPrompt`.

#### Responses API

By default the client calls Chat Completions (`/v1/chat/completions`). `WithResponsesAPI` switches every session of the client to the Responses API (`/v1/responses`):

```go
client, err := openai.New(ctx, apiKey,
    openai.WithModel("gpt-6.1-sol"),
    openai.WithResponsesAPI(),
    openai.WithReasoningEffort("low"),
)
```

Use it for models that accept function tools together with reasoning only on the Responses API.

- Requests are stateless: each call sends the whole conversation with `store: false`, and `previous_response_id` is not used. Session History and history-rewriting middleware work as with Chat Completions.
- The encrypted reasoning items of earlier turns are requested (`include: ["reasoning.encrypted_content"]`), kept in the History, and sent back on the next call. A History saved in Responses mode can be restored into a new session; a History saved with Chat Completions also loads, but its reasoning text is not sent because the Responses API accepts only reasoning items it issued.
- Reasoning summaries, when the response contains them, are returned in `Thoughts`.
- `WithMaxTokens` and the per-call `gollem.WithMaxTokens` are sent as `max_output_tokens`.
- The Responses API has no presence or frequency penalty; `New` returns an error if `WithPresencePenalty` or `WithFrequencyPenalty` is combined with `WithResponsesAPI`.

Chat Completions stays the default because many OpenAI-compatible servers reached through `WithBaseURL` have no `/responses` endpoint.

#### Organization and Base URL

```go
client, err := openai.New(ctx, apiKey,
    openai.WithOrganization("org-id"),
    openai.WithBaseURL("https://custom-endpoint.com"),
)
```

#### Issuer Scope

```go
client, err := openai.New(ctx, apiKey,
    openai.WithIssuerScope("endpoint-a"),
)
```

With Chat Completions, the `reasoning_content` that an OpenAI-compatible server returns is kept in the session history and sent back on later requests, because some servers reject a request with tools that does not send it back. Reasoning content in a history is sent back only to a client with the same model and the same scope. The scope is empty by default. Set different scopes when clients of the same model must not exchange reasoning, for example clients of different endpoints. See [Provider-bound data](history.md#provider-bound-data).

### Environment Variables

- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_ORGANIZATION` - Organization ID (optional)
- `GOLLEM_LOGGING_OPENAI_PROMPT` - Enable prompt logging
- `GOLLEM_LOGGING_OPENAI_RESPONSE` - Enable response logging

## Ollama

The Ollama client talks to an Ollama server through its native API (`/api/chat` and `/api/embed`). It works with a local server, a server elsewhere on the network, and Ollama Cloud.

### Basic Setup

```go
import (
    "context"
    "github.com/gollem-dev/gollem/llm/ollama"
)

// The model must already be pulled to the server: ollama pull qwen3:8b
client, err := ollama.New(ctx, "qwen3:8b")
```

The model name is a required argument because no model is available on every server. The client connects to `http://localhost:11434` (`ollama.DefaultBaseURL`) unless `WithBaseURL` is given.

### Configuration Options

#### Server and Authentication

```go
// A server on another host
client, err := ollama.New(ctx, "qwen3:8b",
    ollama.WithBaseURL("http://gpu-box.internal:11434"),
)

// Ollama Cloud: create an API key on ollama.com
client, err := ollama.New(ctx, "gemma4:31b",
    ollama.WithBaseURL("https://ollama.com"),
    ollama.WithAPIKey(apiKey), // sent as "Authorization: Bearer <apiKey>"
)

// A custom HTTP client (proxy, TLS settings)
client, err := ollama.New(ctx, "qwen3:8b",
    ollama.WithHTTPClient(&http.Client{Transport: transport}),
)
```

The default HTTP client has no timeout, so a long streamed response is not cut off. Cancel a request through its context instead.

#### Context Window

```go
client, err := ollama.New(ctx, "qwen3:8b",
    ollama.WithNumCtx(32768),
)
```

Set `WithNumCtx` to the context size your conversations need. The server default is often much smaller than what the model supports. By default Ollama drops the oldest messages without reporting it when a prompt exceeds the context size; gollem disables that (`truncate: false`, `shift: false`), so the request fails instead and the error is tagged with `gollem.ErrTagTokenExceeded`. The `middleware/compacter` middleware uses that tag to summarize the history and retry, the same as with the other providers.

#### Generation Parameters

```go
client, err := ollama.New(ctx, "qwen3:8b",
    ollama.WithTemperature(0.7), // options.temperature
    ollama.WithTopP(0.9),        // options.top_p
    ollama.WithTopK(40),         // options.top_k
    ollama.WithMaxTokens(2048),  // options.num_predict
)
```

Parameters that are not set are not sent, so the model defaults apply. `gollem.WithMaxTokens` overrides `num_predict` for a single `Generate` or `Stream` call. Temperature and top-p cannot be changed per call; to use different values, create another client with different options.

#### Thinking

```go
// Turn thinking on or off for models that support it
client, err := ollama.New(ctx, "qwen3:8b", ollama.WithThink(false))

// Or pick a named level, for a model that defines levels
client, err := ollama.New(ctx, modelWithLevels, ollama.WithThinkLevel("high"))
```

Use a level listed in `thinking.values` of the server's `/api/show` response for the model. The client does not validate the level. Depending on the model, the server either rejects an unsupported level with an error or silently applies the model default, so check the value against `/api/show` before using it. Without either option the model default applies. Thinking output is returned in `Response.Thoughts`.

#### Keep Alive, System Prompt and Embeddings

```go
client, err := ollama.New(ctx, "qwen3:8b",
    ollama.WithKeepAlive(30*time.Minute),        // a negative value keeps the model loaded indefinitely
    ollama.WithSystemPrompt("You are concise."),  // used when the session sets no system prompt
    ollama.WithEmbeddingModel("nomic-embed-text"), // required for GenerateEmbedding
)
```

#### Issuer Scope

```go
client, err := ollama.New(ctx, "qwen3:8b",
    ollama.WithIssuerScope("gpu-box"),
)
```

Thinking in a history is sent back only to a client with the same model and the same scope. The scope is empty by default. Set different scopes when clients of different servers use the same model name for different models. See [Provider-bound data](history.md#provider-bound-data).

### Differences from Other Providers

- **Token counting**: Ollama has no API that counts tokens, so `Session.CountToken` returns an error wrapping `gollem.ErrUnsupportedOperation`.
- **`gollem.WithToolCallsDisabled`**: Ollama has no parameter that forbids tool calls, so the tool definitions are left out of that call.
- **Tool parameter constraints**: the server reads only `type`, `description`, `enum`, `items`, `properties`, `required` and `anyOf` of each tool parameter. Constraints such as `minimum`, `pattern` or `additionalProperties` are sent but ignored.
- **Structured outputs**: a response schema is sent as `format`. Ollama Cloud does not support structured outputs.
- **Input types**: images are sent as base64 data. An image given only by URL and PDF input return `gollem.ErrInvalidParameter`.
- **Prompt caching**: `gollem.WithSessionPromptCache` has no effect. The server reuses its own KV cache, and the cached prompt tokens are reported in `Response.CacheReadInputToken`.

### Environment Variables

The client reads no environment variables. The integration tests use:

- `TEST_OLLAMA_MODEL` - Chat model to test with; the tests are skipped when it is not set
- `TEST_OLLAMA_BASE_URL` - Server address (optional, default `http://localhost:11434`)
- `TEST_OLLAMA_EMBEDDING_MODEL` - Embedding model for the embedding test (optional)

## PDF Input Support

gollem supports sending PDF documents to LLMs as input, enabling document analysis, extraction, and summarization.

### Creating PDF Input

```go
// From byte data
data, err := os.ReadFile("document.pdf")
if err != nil {
    return err
}
pdf, err := gollem.NewPDF(data)
if err != nil {
    return err
}

// From io.Reader
f, err := os.Open("document.pdf")
if err != nil {
    return err
}
defer f.Close()
pdf, err := gollem.NewPDFFromReader(f)
if err != nil {
    return err
}

// With custom max size
pdf, err := gollem.NewPDFFromReader(f, gollem.WithMaxPDFSize(64*1024*1024)) // 64MB
```

### Sending PDF to LLM

```go
result, err := session.Generate(ctx, []gollem.Input{
    pdf,
    gollem.Text("What are the key findings in this document?"),
})
if err != nil {
    return err
}
fmt.Println(result.Texts)
```

### Provider Compatibility

| Provider | PDF Support | Implementation |
|----------|------------|----------------|
| Claude (Anthropic) | Yes | Document block with base64-encoded data |
| Claude (Vertex AI) | Yes | Document block with base64-encoded data |
| Gemini | Yes | Inline data with `application/pdf` MIME type |
| OpenAI (Chat Completions) | No | Chat Completions does not accept PDF via the image_url field |
| OpenAI (Responses API) | Yes, with a model that accepts PDF input | Sent as an `input_file` item with base64-encoded PDF data (`openai.WithResponsesAPI()`) |
| Ollama | No | Ollama has no PDF input; a PDF input or a history containing one returns `gollem.ErrInvalidParameter` |

### Validation and Safety

- **Format validation**: PDF data must start with the `%PDF-` magic bytes
- **Size limit**: Default maximum is 32MB (`gollem.DefaultMaxPDFSize`), configurable via `gollem.WithMaxPDFSize()`
- **Memory protection**: `NewPDFFromReader` uses `io.LimitReader` internally to prevent reading unlimited data from untrusted sources

### History Round-Trip

PDF inputs are preserved during cross-provider history conversion. A PDF sent to Claude can be restored when converting history to Gemini format, and vice versa. With OpenAI, a PDF in the History is sent as an `input_file` item in Responses API mode; the Chat Completions path sends it as a `data:application/pdf;base64,...` URL in an `image_url` part, which Chat Completions does not accept.

## Common Configuration Patterns

### Session Configuration

All LLM clients support common session options:

```go
session, err := client.NewSession(ctx,
    gollem.WithSessionHistory(history),
    gollem.WithSessionContentType(gollem.ContentTypeJSON),
    gollem.WithSessionTools(tool1, tool2),
    gollem.WithSessionSystemPrompt("You are a helpful assistant."),
)
```

### Per-Call Options

Override session defaults for a single `Generate` or `Stream` call:

```go
resp, err := session.Generate(ctx, inputs,
    gollem.WithMaxTokens(256),
    gollem.WithGenerateResponseSchema(schema), // forces JSON output for this call
)
```

See [Per-Call Generate Options](schema.md#per-call-generate-options) for details.

### Embedding Generation

Providers that support embeddings (OpenAI, Gemini and Ollama):

```go
embeddings, err := client.GenerateEmbedding(ctx, 
    768,           // dimension
    []string{      // texts to embed
        "Hello world",
        "Another text",
    },
)
```

### Reporting the Configured Model Name

A caller that meters, prices, or audits generations needs to record which model
produced a response. `gollem.ModelNamer` is an optional interface that reports
it, so the caller no longer has to mirror its own client-construction settings:

```go
type ModelNamer interface {
    Model() string
}
```

All clients shipped with gollem implement it — `claude.Client`,
`claude.VertexClient`, `gemini.Client` and `openai.Client`. Obtain the name with
a type assertion on the `gollem.LLMClient` you already hold:

```go
client, err := openai.New(ctx, apiKey, openai.WithModel("gpt-5-mini"))
if err != nil {
    return err
}

var llm gollem.LLMClient = client

model := "unknown"
if namer, ok := llm.(gollem.ModelNamer); ok {
    model = namer.Model() // "gpt-5-mini"
}
```

The interface is optional: `LLMClient` itself is unchanged, so a custom client
or a mock that does not implement `ModelNamer` keeps working, and the assertion
simply reports `false`.

`Model()` returns the name the client was **configured** with — the value passed
to `WithModel` (or `WithVertexModel`), or the provider default when no option
was given. It is not the model id an API response may report: an alias can
resolve to a dated snapshot, and returning that would break a caller keying a
price table by the name it configured. The value is fixed at construction time,
so calling `Model()` performs no API request and is safe to call concurrently.

### Finish Reason

`Response.FinishReason` holds the reason the provider gave for ending the
generation, as the provider returned it. gollem does not map it to its own
values, so compare it with the provider's documented strings:

| Provider | Source field | Example values |
|---|---|---|
| Claude (Anthropic, Vertex AI) | `stop_reason` | `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`, `refusal` |
| Gemini | `finishReason` of the candidate | `STOP`, `MAX_TOKENS`, `SAFETY` |
| OpenAI (Chat Completions) | `finish_reason` of the first choice | `stop`, `length`, `tool_calls`, `content_filter` |
| OpenAI (Responses API) | `incomplete_details.reason`, or `status` when the response has no incomplete details | `max_output_tokens`, `content_filter`, `completed` |
| Ollama | `done_reason` | `stop`, `length` |

The field is empty when the provider returned no value. For Claude,
`Response.StopSequence` holds the custom stop sequence that ended the
generation, when there is one.

`Session.Generate` and `Session.Stream` do not return an error because the
provider reports a refusal or a truncated response; the caller decides how to
handle it. Errors that existed before are still returned, for example when the
arguments of a tool call cut off by the token limit cannot be decoded, or when
Gemini reports `PROHIBITED_CONTENT`. `Query` and `SessionQuery` return an error
for a refusal (see [Refusal Details](#refusal-details)). Check `err` first,
then `FinishReason`. This matters for a structured query through `Generate`,
where an empty `Texts` alone does not tell why no JSON came back:

```go
resp, err := session.Generate(ctx, input, gollem.WithGenerateResponseSchema(schema))
if err != nil {
    return err
}
if len(resp.Texts) == 0 {
    switch resp.FinishReason {
    case "refusal":
        return errors.New("the model refused the request")
    case "max_tokens":
        return errors.New("the response was cut off by max_tokens")
    }
}
```

Claude also reports, in `Response.OtherBlockTypes`, the type of every content
block other than `text` and `tool_use`, in the order received (for example
`thinking`, `redacted_thinking`, `server_tool_use`). Only the type is recorded;
the text of thinking blocks is in `Response.Thoughts`. A response with no
`Texts` and only a `redacted_thinking` block can therefore be told apart from a
response with no content blocks at all. Other providers leave the field empty.

With `Stream`, the finish reason and the stop sequence are set on the response
for the event that carried them, and each block type on the response for the
event that started the block. Such a response can have no texts, thoughts or
function calls. For the OpenAI Responses API and Ollama, the finish reason
arrives with the final token counts and is set on the last response.

### Refusal Details

When the provider refuses the request, or blocks the prompt or the output,
`Response.Refusal` holds the details the provider gave. It is `nil` otherwise.
The values are passed through as the provider returned them:

| Provider | Set when | `Reason` | `Categories` | `Explanation` |
|---|---|---|---|---|
| Claude (Anthropic, Vertex AI) | `stop_reason` is `refusal` | `refusal` | `stop_details.category`, for example `cyber`, `bio`, `frontier_llm`, `reasoning_extraction`, `general_harms` | `stop_details.explanation` |
| Gemini | the candidate's `finishReason` is `SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `IMAGE_SAFETY`, `IMAGE_PROHIBITED_CONTENT` or `IMAGE_RECITATION` | that `finishReason` | the `category` of each safety rating marked `blocked` | `finishMessage` |
| Gemini | `promptFeedback.blockReason` is set (the prompt was blocked and no candidate was returned) | `blockReason` | the `category` of each prompt safety rating marked `blocked` | `blockReasonMessage` |
| OpenAI (Chat Completions) | the message has `refusal`, or `finish_reason` is `content_filter` | `content_filter` when `finish_reason` is `content_filter`, otherwise `refusal` | always empty | the `refusal` message |
| OpenAI (Responses API) | the output has a `refusal` content part, or `incomplete_details.reason` is `content_filter` | `content_filter` when `incomplete_details.reason` is `content_filter`, otherwise `refusal` | always empty | the text of the `refusal` content parts |
| Ollama | never; the API reports no refusal | | | |

`Categories` and `Explanation` are empty when the provider gave no value; for
example, Claude returns no explanation for some categories. The explanation
text is not guaranteed to be stable, so do not compare it with fixed strings.
For a Gemini prompt block, `FinishReason` is empty because no candidate was
returned. The OpenAI Responses API also returns the refusal message in `Texts`,
as before.

```go
resp, err := session.Generate(ctx, input)
if err != nil {
    return err
}
if r := resp.Refusal; r != nil {
    slog.Warn("the provider refused the request",
        "reason", r.Reason, "categories", r.Categories, "explanation", r.Explanation)
}
```

With `Stream`, the refusal is set on the response for the event that carried
it: for Claude, the event with the stop reason; for OpenAI Chat Completions,
the chunk with the finish reason; for the OpenAI Responses API, the last
response; for Gemini, the chunk that reported the block.

`Query` and `SessionQuery` return an error that wraps
`gollem.ErrProhibitedContent` when the response has a refusal, without retrying,
because a correction prompt does not resolve a refusal. The error carries the
details as goerr values `finish_reason`, `refusal_reason`, `refusal_categories`
and `refusal_explanation`. The Gemini error for `PROHIBITED_CONTENT`, which
also wraps `gollem.ErrProhibitedContent`, carries `refusal_reason`,
`refusal_categories` and `refusal_explanation`:

```go
_, err := gollem.Query[Answer](ctx, client, prompt)
if errors.Is(err, gollem.ErrProhibitedContent) {
    values := goerr.Values(err)
    log.Printf("refused: category=%v explanation=%v",
        values["refusal_categories"], values["refusal_explanation"])
}
```

### Error Handling

All providers return standardized errors that can be checked:

```go
resp, err := session.Generate(ctx, []gollem.Input{input})
if err != nil {
    // Check for specific error types
    // Handle token limit errors, rate limits, etc.
    return err
}
```

## Debugging and Monitoring

### Enable Logging

Use the `trace/logger` package to enable detailed logging for LLM interactions:

```go
import tracelogger "github.com/gollem-dev/gollem/trace/logger"

// Log LLM requests and responses
handler := tracelogger.New(
    tracelogger.WithEvents(tracelogger.LLMRequest, tracelogger.LLMResponse),
)

agent := gollem.New(client, gollem.WithTraceHandler(handler))
```

See [debugging.md](debugging.md) for full details on available events and configuration.

### Log Output Format

Logs are structured via `slog`:

```json
{
  "level": "INFO",
  "msg": "llm_call_end",
  "elapsed_ms": 1234,
  "texts": ["Generated response text"]
}
```