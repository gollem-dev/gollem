# History Management

History represents a conversation history that can be used across different LLM sessions. It stores messages in a provider-independent format (`Message` with typed `MessageContent`), and its `LLType` field records the provider that created it (OpenAI, Claude, Gemini, or Ollama).

## Automatic vs Manual History Management

### Automatic History Management (Recommended)

The `Execute` method provides automatic session management, eliminating the need for manual history handling:

```go
agent := gollem.New(client, gollem.WithTools(tools...))

// First interaction - creates new session automatically
err := agent.Execute(ctx, "Hello, I'm working on a project.")

// Follow-up - automatically remembers previous context
err = agent.Execute(ctx, "Can you help me with the next step?")

// Access conversation history if needed
history := agent.Session().History()
messageCount := history.ToCount()
```

**Benefits:**
- No manual history management required
- Conversation context preserved automatically
- Simplified API for conversational applications
- Reduced boilerplate code

### Manual History Management (Legacy)

For backward compatibility and advanced use cases, manual history management is still supported:

```go
// Legacy approach using Prompt method
var history *gollem.History

newHistory, err := agent.Prompt(ctx, "Hello", gollem.WithHistory(history))
if err != nil {
    return err
}
history = newHistory

// Continue conversation with manual history
newHistory, err = agent.Prompt(ctx, "Continue", gollem.WithHistory(history))
```

## Version Management

History includes version information to ensure compatibility. The current version is **4**.

### Version History

| Version | Changes |
|---------|---------|
| 1 | Initial format with provider-specific message dialects |
| 2 | Introduced unified message format across providers |
| 3 | Removed legacy function call fields and provider-specific dialects. Messages use a single canonical representation (`Message` with `MessageContent` typed data) |
| 4 | Replaced `MessageContent.Meta` with `MessageContent.Provider`, which records the issuer of provider-bound data such as thinking signatures. A Gemini part that carries only a thought signature is stored as a thinking content with empty text instead of an empty text content |

### Compatibility

- **Migration from an earlier version is not supported.** History serialized with v1, v2 or v3 cannot be deserialized into v4. A v3 history has no record of which model and scope issued its signatures, so it cannot be converted correctly.
- If you have persisted histories of an earlier version, discard them or re-create the conversations with the current library version.
- Version is stored in the `"version"` JSON field of the serialized `History` struct. When deserializing, callers should verify that the version matches `gollem.HistoryVersion` before use.
- Future versions will document migration paths when feasible.
- The serialized format is not covered by semver guarantees. See [Compatibility Policy](compatibility.md#serialized-history-format).

## Session Persistence

History is essential for maintaining conversation context across stateless sessions. Common use cases include:

### Backend Services
- **Stateless HTTP requests**: When your backend service receives requests from different instances or after restarts
- **Multiple API calls**: When you need to maintain conversation context across multiple API calls
- **Load balancing**: When sessions may be handled by different instances

### Distributed Systems
- **Microservices**: When conversations need to be shared across different services
- **Horizontal scaling**: When you need to load balance conversations across multiple servers
- **Service restarts**: When conversations need to be resumed after service restarts

### Long-running Conversations
- **Session resumption**: When conversations need to be resumed after service restarts
- **Conversation history**: When implementing features like "continue previous conversation"
- **Multi-session workflows**: When users switch between different devices or sessions

## Portability

History can be easily serialized/deserialized using standard JSON marshaling. This enables:

### Storage Options
- **Database persistence**: Store conversations in SQL or NoSQL databases
- **File storage**: Save conversations to local or cloud file systems
- **Cache systems**: Use Redis or Memcached for temporary storage
- **Message queues**: Transfer conversations through messaging systems

### Use Cases
- **Conversation backup**: Backup important conversations for disaster recovery
- **Analytics**: Analyze conversation patterns and user behavior
- **Audit trails**: Maintain records for compliance and debugging
- **Cross-platform sync**: Synchronize conversations across different platforms

## LLM Type Compatibility

A session accepts a History created by any provider; the clients do not check `LLType`. Each client converts the messages to its own API format, so the content must be something the destination can send. A client can reject content its API cannot represent. For example, the Ollama client returns an error from `NewSession` when the history contains a PDF or an image given only by URL.

### Provider-bound data

Some providers return data that only they can interpret and that must be sent back unchanged, such as Claude thinking signatures and redacted thinking, and Gemini thought signatures. The history stores this data in `MessageContent.Provider` together with its issuer:

```go
type Issuer struct {
    Provider LLMType // the provider of the client that created the data
    Model    string  // the model the client was configured with
    Scope    string  // a value set with the client's WithIssuerScope option
}
```

Two issuers are the same only when all three fields are equal. The built-in clients record their issuer on every thinking content, including the reasoning of OpenAI and Ollama, which has no signature. The Gemini client also records it on text, tool call, image and PDF content that carries a thought signature. The Claude client also records it on every tool call the API returned, together with the arguments JSON as the API returned it. When the history is sent back to the same issuer and that JSON still decodes to the tool call's `Arguments`, the client sends that JSON instead of encoding `Arguments`, so the keys keep the order the API returned. The Anthropic SDK still applies its own string escaping when it sends the request, so the bytes can differ from the API response. When `Arguments` have been edited, or the destination is another issuer, the client encodes `Arguments`, which sorts the keys alphabetically.

Before a client converts a history to its API format, it applies `gollem.FilterProviderData` with its own issuer as the destination:

| Content | Provider-bound data | Sent to the destination |
|---------|---------------------|-------------------------|
| thinking | issued by the destination | as is, with its data |
| thinking | issued by another issuer, or none | not sent |
| any other type | issued by the destination | as is, with its data |
| any other type | issued by another issuer | without its data |
| any other type | none | as is |

A message left with no content is not sent. `FilterProviderData` does not modify the history.

As a result, a session receives thinking only from a client with the same provider, model and scope. When you switch the model or the provider of a conversation, the earlier thinking is not sent, while the text, tool calls and tool results are. A thinking content you create with `gollem.NewThinkingContent` has no issuer and is not sent to any provider.

### Setting an issuer scope

The scope is empty unless you set it, so a single account needs no configuration. Set a scope when two clients of the same provider and model must not exchange provider-bound data, because the provider would reject it. For example, a Claude signature created with one Anthropic account is rejected by the API when a session of another account sends it:

```go
clientA, _ := claude.New(ctx, keyA, claude.WithModel(model), claude.WithIssuerScope("tenant-a"))
clientB, _ := claude.New(ctx, keyB, claude.WithModel(model), claude.WithIssuerScope("tenant-b"))

sessionA, _ := clientA.NewSession(ctx)
// ... generate with sessionA ...
history, _ := sessionA.History()

// The thinking of sessionA is not sent; text and tool calls are.
sessionB, _ := clientB.NewSession(ctx, gollem.WithSessionHistory(history))
```

A client created with `claude.New` and one created with `claude.NewWithVertex` record the same provider. Whether Vertex AI accepts thinking signatures issued through the Anthropic API, and the reverse, has not been confirmed. To keep their data apart, set different scopes:

```go
anthropicClient, _ := claude.New(ctx, apiKey, claude.WithIssuerScope("anthropic"))
vertexClient, _ := claude.NewWithVertex(ctx, region, projectID, claude.WithVertexIssuerScope("vertex:"+projectID))
```

### Implementing your own client

A custom `LLMClient` and `Session` follow the same rules as the built-in clients when they do the following:

1. When converting an API response to a history, set `MessageContent.Provider` on every content that carries provider-bound data, with the session's issuer. Set it on every thinking content, even without data, so that the thinking is sent back only to your client.
2. Before converting a history to your API format, call `gollem.FilterProviderData` with the session's issuer.
3. Use a `Provider` value other than the `LLMType` constants of the built-in clients. With the same provider, model and an empty scope, the data your client records is sent by the built-in client to its API. Use a built-in value only when your client is meant to exchange provider-bound data with that built-in client.
4. Store data that belongs to no text or tool call, such as an opaque block returned on its own, on a thinking content with empty text. `FilterProviderData` then removes it for other issuers, and it adds no element to `Response.Thoughts`.
5. Do not add your own `MessageContentType`. `FilterProviderData` keeps content of a type other than thinking even when another issuer created it, and a built-in client returns an error or skips content of a type it does not know.

```go
issuer := gollem.Issuer{Provider: "my-provider", Model: s.model, Scope: s.scope}

// API response -> history
c, err := gollem.NewThinkingContent(block.Text)
if err != nil {
    return nil, err
}
c.Provider = &gollem.ProviderData{Issuer: issuer, Data: block.RawSignature} // any JSON

// history -> API request
messages := gollem.FilterProviderData(history.Messages, issuer)
```

## Usage Guidelines

### With Automatic Session Management (Recommended)

```go
// Create agent with automatic session management
agent := gollem.New(client,
    gollem.WithTools(tools...),
    gollem.WithSystemPrompt("You are a helpful assistant."),
)

// Execute multiple interactions - history managed automatically
err := agent.Execute(ctx, "What's the weather like?")
err = agent.Execute(ctx, "What about tomorrow?") // Remembers previous context

// Access history when needed
if history := agent.Session().History(); history != nil {
    messageCount := history.ToCount()
    fmt.Printf("Conversation has %d messages\n", messageCount)
    
    // Serialize for storage
    data, err := json.Marshal(history)
    if err != nil {
        return fmt.Errorf("failed to marshal history: %w", err)
    }
    
    // Store in database, file, etc.
    err = saveToDatabase(data)
}
```

### With Manual History Management (Legacy)

1. **Get History from Prompt response:**
   ```go
   // Create a new gollem agent
   agent := gollem.New(client)

   // Get response from Prompt
   history, err := agent.Prompt(ctx, "What is the weather?")
   if err != nil {
       return nil, fmt.Errorf("failed to get prompt response: %w", err)
   }
   ```

2. **Store the History for future use:**
   ```go
   // Store history in your database or storage
   jsonData, err := json.Marshal(history)
   if err != nil {
       return fmt.Errorf("failed to marshal history: %w", err)
   }
   
   // Save to your preferred storage
   err = database.SaveConversation(userID, jsonData)
   ```

3. **Use stored History in a new session:**
   ```go
   // Restore history
   jsonData, err := database.LoadConversation(userID)
   if err != nil {
       return fmt.Errorf("failed to load conversation: %w", err)
   }
   
   var restoredHistory gollem.History
   if err := json.Unmarshal(jsonData, &restoredHistory); err != nil {
       return fmt.Errorf("failed to unmarshal history: %w", err)
   }

   // Use history in next Prompt call
   newHistory, err := agent.Prompt(ctx, "What about tomorrow?", gollem.WithHistory(&restoredHistory))
   if err != nil {
       return nil, fmt.Errorf("failed to get prompt response: %w", err)
   }
   ```

Note: The History returned from Prompt contains the complete conversation history, so there's no need to manage or track individual messages. Each Prompt response provides a new History instance that includes all previous messages.

## Automatic History Persistence with HistoryRepository

`HistoryRepository` is an interface that lets gollem automatically load and save conversation history to any storage backend — filesystem, S3, GCS, a database, etc.

```go
type HistoryRepository interface {
    Load(ctx context.Context, sessionID string) (*History, error)
    Save(ctx context.Context, sessionID string, history *History) error
}
```

### How it works

- **On first `Execute`**: history is loaded from the repository using `sessionID`. If no history exists yet, the session starts fresh.
- **After each LLM round-trip**: history is saved automatically. This ensures that even if the process crashes mid-conversation, progress up to the last completed round-trip is preserved.
- `Load` returns `nil, nil` when the session ID is not found (new session — not an error).
- `Save` always overwrites the previous value for that session ID.

### Usage

```go
agent := gollem.New(client,
    gollem.WithHistoryRepository(repo, "user-123"),
)

// First run: loads history from repo (or starts fresh if none exists)
resp, err := agent.Execute(ctx, gollem.Text("Hello!"))

// Second run (same agent): session already exists, no Load is called again
resp, err = agent.Execute(ctx, gollem.Text("What did I just say?"))
```

> **Note**: `WithHistory` and `WithHistoryRepository` cannot be used together — an error is returned from `Execute` if both are set.

### Implementing HistoryRepository

The interface is intentionally minimal. A filesystem implementation looks like this (see also [examples/history](../examples/history/main.go)):

```go
type FileRepository struct{ dir string }

func (r *FileRepository) Load(ctx context.Context, id string) (*gollem.History, error) {
    data, err := os.ReadFile(filepath.Join(r.dir, id+".json"))
    if errors.Is(err, os.ErrNotExist) {
        return nil, nil
    }
    if err != nil {
        return nil, err
    }
    var h gollem.History
    return &h, json.Unmarshal(data, &h)
}

func (r *FileRepository) Save(ctx context.Context, id string, h *gollem.History) error {
    data, _ := json.Marshal(h)
    return os.WriteFile(filepath.Join(r.dir, id+".json"), data, 0600)
}
```

For cloud storage, implement the same two methods using your SDK of choice — gollem imposes no additional constraints.

## Best Practices

### Prefer HistoryRepository over manual JSON marshaling

For any application that persists history across requests, `WithHistoryRepository` is the recommended approach. It removes boilerplate, guarantees saves happen after every round-trip (even on mid-conversation failures), and keeps session lifetime management out of your application code.

Use manual `json.Marshal` / `json.Unmarshal` only when you need one-off exports (e.g., backup, analytics, or migration).

### Choose a stable session ID

The session ID is the primary key for history. Use an ID that is stable for the lifetime of the conversation — for example a user ID, a ticket ID, or a UUID generated when the conversation starts. Avoid IDs that change between requests (e.g., request IDs).

### Validate the history version before restoring

`History.UnmarshalJSON` returns `ErrHistoryVersionMismatch` if the serialized version does not match `gollem.HistoryVersion`. When loading from persistent storage, handle this error explicitly and start a fresh session rather than crashing:

```go
var h gollem.History
if err := json.Unmarshal(data, &h); err != nil {
    if errors.Is(err, gollem.ErrHistoryVersionMismatch) {
        // Stored history is from an older version — start fresh
        return nil, nil
    }
    return nil, err
}
return &h, nil
```

`HistoryRepository.Load` implementations should apply the same pattern.

### Check the content before switching providers

A history can be restored into a session of another provider, but only the content that provider supports can be sent, and thinking is sent only to the client that created it (see [LLM Type Compatibility](#llm-type-compatibility)). Test the switch with histories that contain the content types your application uses, such as images, PDFs, and thinking blocks.

## Next Steps

- Learn more about [tool creation](tools.md)
- Explore [MCP server integration](mcp.md)
- Check out [practical examples](examples.md)
- Review the [getting started guide](getting-started.md)
- Explore the [complete documentation](README.md)

