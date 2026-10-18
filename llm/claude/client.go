package claude

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/internal/convert"
	"github.com/gollem-dev/gollem/internal/jsonutil"
	"github.com/gollem-dev/gollem/internal/schema"
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/goerr/v2"
	"github.com/m-mizutani/jsonex"
)

// generationParameters represents the parameters for text generation.
type generationParameters struct {
	// Temperature controls randomness in the output.
	// Higher values make the output more random, lower values make it more focused.
	Temperature float64

	// TopP controls diversity via nucleus sampling.
	// Higher values allow more diverse outputs.
	TopP float64

	// MaxTokens limits the number of tokens to generate.
	MaxTokens int64

	// maxTokensSet records whether the caller chose MaxTokens. The constructor
	// resolves the model's documented maximum only when it did not, so an
	// explicit value is always sent as given, including an invalid one.
	maxTokensSet bool

	// effort is sent as output_config.effort. Empty means not sent, so the
	// model default applies.
	effort Effort

	// adaptiveThinking records whether the caller chose adaptive thinking.
	// When false, no thinking is sent and the model default applies.
	adaptiveThinking bool

	// thinkingDisplay is sent as thinking.display when adaptiveThinking is
	// set. Empty means not sent.
	thinkingDisplay ThinkingDisplay
}

// ThinkingDisplay is the display sent as thinking.display with adaptive
// thinking. It controls whether the thinking text is returned. See
// https://platform.claude.com/docs/en/build-with-claude/thinking
type ThinkingDisplay string

const (
	// ThinkingDisplaySummarized returns a summary of the thinking text.
	ThinkingDisplaySummarized ThinkingDisplay = "summarized"
	// ThinkingDisplayOmitted returns thinking blocks with empty text and a signature.
	ThinkingDisplayOmitted ThinkingDisplay = "omitted"
)

// Effort is the effort level sent as output_config.effort. It controls how many
// tokens Claude spends on a response, including text, tool calls and thinking.
// Which levels a model accepts depends on the model; see
// https://platform.claude.com/docs/en/build-with-claude/effort
type Effort string

const (
	// EffortLow spends the fewest tokens.
	EffortLow Effort = "low"
	// EffortMedium balances token usage and capability.
	EffortMedium Effort = "medium"
	// EffortHigh spends as many tokens as the task needs.
	EffortHigh Effort = "high"
	// EffortXHigh extends EffortHigh for long-running agentic work.
	EffortXHigh Effort = "xhigh"
	// EffortMax places no constraint on token spending.
	EffortMax Effort = "max"
)

// defaultNonStreamingTimeout is the request timeout attached to every
// non-streaming Messages call.
//
// It matches the SDK's own default for such requests, so the effective timeout
// is unchanged. Supplying it explicitly is what matters: the SDK otherwise
// derives a timeout from max_tokens and refuses to send anything it estimates
// at over 10 minutes, which rejects the model ceilings resolved by
// resolveMaxOutputTokens before the request leaves the process.
// See anthropic-sdk-go CalculateNonStreamingTimeout.
const defaultNonStreamingTimeout = 10 * time.Minute

// setTemperatureAndTopP sets temperature and/or top_p on the request params.
// Claude does not allow both to be specified simultaneously.
// Returns an error if both are set.
func setTemperatureAndTopP(params *anthropic.MessageNewParams, temperature, topP float64) error {
	if temperature >= 0 && topP >= 0 {
		return goerr.New("both Temperature and TopP are set; Claude does not allow both")
	}
	if temperature >= 0 {
		params.Temperature = anthropic.Float(temperature)
	} else if topP >= 0 {
		params.TopP = anthropic.Float(topP)
	}
	return nil
}

// Client is a client for the Claude API.
// It provides methods to interact with Anthropic's Claude models.
type Client struct {
	// client is the underlying Claude client.
	client *anthropic.Client

	// defaultModel is the model to use for chat completions.
	// It can be overridden using WithModel option.
	defaultModel string

	// apiKey is the API key for authentication.
	apiKey string

	// baseURL is the custom base URL for the Claude API.
	// If empty, uses the default Anthropic API endpoints.
	baseURL string

	// generation parameters
	params generationParameters

	// systemPrompt is the system prompt to use for chat completions.
	systemPrompt string

	// timeout for API requests
	timeout time.Duration

	// issuerScope distinguishes this client's provider-bound data from data
	// issued by another client of the same model. See WithIssuerScope.
	issuerScope string
}

// Option is a function that configures a Client.
type Option func(*Client)

// WithModel sets the default model to use for chat completions.
// The model name should be a valid Claude model identifier.
// Default: anthropic.ModelClaude3_5SonnetLatest
func WithModel(modelName string) Option {
	return func(c *Client) {
		c.defaultModel = modelName
	}
}

// WithTemperature sets the temperature parameter for text generation.
// Higher values make the output more random, lower values make it more focused.
// Range: 0.0 to 1.0
// Default: 0.7
func WithTemperature(temp float64) Option {
	return func(c *Client) {
		c.params.Temperature = temp
	}
}

// WithTopP sets the top_p parameter for text generation.
// Controls diversity via nucleus sampling.
// Range: 0.0 to 1.0
// Default: 1.0
func WithTopP(topP float64) Option {
	return func(c *Client) {
		c.params.TopP = topP
	}
}

// WithMaxTokens sets the maximum number of tokens to generate.
// When not set, the model's documented maximum output tokens is used.
// A value the API does not accept, such as zero or one above the model's
// limit, is sent as given and rejected by the API.
func WithMaxTokens(maxTokens int64) Option {
	return func(c *Client) {
		c.params.MaxTokens = maxTokens
		c.params.maxTokensSet = true
	}
}

// WithEffort sets the effort level sent as output_config.effort on every
// request of the client's sessions. When not set, no effort is sent and the
// model default applies. A level the model does not support is sent as given
// and rejected by the API.
func WithEffort(effort Effort) Option {
	return func(c *Client) {
		c.params.effort = effort
	}
}

// WithAdaptiveThinking sends thinking {"type":"adaptive","display":display}
// on every request of the client's sessions. An empty display sends
// {"type":"adaptive"} and leaves the display to the model default. When not
// set, no thinking is sent and the model default applies. A model that does
// not accept adaptive thinking rejects the request.
func WithAdaptiveThinking(display ThinkingDisplay) Option {
	return func(c *Client) {
		c.params.adaptiveThinking = true
		c.params.thinkingDisplay = display
	}
}

// WithTimeout sets the timeout for API requests
func WithTimeout(timeout time.Duration) Option {
	return func(c *Client) {
		c.timeout = timeout
	}
}

// WithSystemPrompt sets the system prompt for the client
func WithSystemPrompt(prompt string) Option {
	return func(c *Client) {
		c.systemPrompt = prompt
	}
}

// WithBaseURL sets the custom base URL for the Claude API.
// Allows usage with compatible endpoints, proxies, or self-hosted instances.
// If empty, uses the default Anthropic API endpoints.
func WithBaseURL(url string) Option {
	return func(c *Client) {
		c.baseURL = url
	}
}

// WithIssuerScope sets the scope recorded on the provider-bound data this
// client creates, such as thinking signatures. Data is sent back only to a
// session whose client has the same model and the same scope. Set a distinct
// scope for each Anthropic account, so that a History created with one account
// does not send its signatures to another account, which the API rejects.
// Default: "" (empty).
func WithIssuerScope(scope string) Option {
	return func(c *Client) {
		c.issuerScope = scope
	}
}

// New creates a new client for the Claude API.
// It requires an API key and can be configured with additional options.
func New(ctx context.Context, apiKey string, options ...Option) (*Client, error) {
	client := &Client{
		defaultModel: "claude-sonnet-4-5-20250929",
		apiKey:       apiKey,
		baseURL:      "", // Default empty, will be set by options
		params: generationParameters{
			Temperature: -1.0, // -1 indicates not set (0.0 is valid)
			TopP:        -1.0, // -1 indicates not set (0.0 is valid)
		},
		timeout: 30 * time.Second, // Default timeout
	}

	for _, option := range options {
		option(client)
	}

	// The Messages API requires max_tokens, so it cannot be omitted. Fill in
	// the model's documented ceiling only when the caller did not choose one.
	if !client.params.maxTokensSet {
		client.params.MaxTokens = resolveMaxOutputTokens(client.defaultModel)
	}

	clientOptions := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}

	// Add BaseURL if specified
	if client.baseURL != "" {
		clientOptions = append(clientOptions, option.WithBaseURL(client.baseURL))
	}

	// Add timeout if specified
	if client.timeout > 0 {
		httpClient := &http.Client{
			Timeout: client.timeout,
		}
		clientOptions = append(clientOptions, option.WithHTTPClient(httpClient))
	}

	newClient := anthropic.NewClient(clientOptions...)
	client.client = &newClient

	return client, nil
}

// Session is a session for the Claude chat.
// It maintains the conversation state and handles message generation.
type Session struct {
	// apiClient is the API client interface for dependency injection.
	apiClient apiClient

	// defaultModel is the model to use for chat completions.
	defaultModel string

	// tools are the available tools for the session.
	tools []anthropic.ToolUnionParam

	// historyMessages maintains history in Claude native format for efficiency
	historyMessages []anthropic.MessageParam

	// issuer identifies this session as the issuer of provider-bound data.
	issuer gollem.Issuer

	// generation parameters
	params generationParameters

	cfg gollem.SessionConfig
}

// Model returns the model name this client generates through. It is the name
// the client was configured with, so a caller can key its own tables by the
// same string it passed to WithModel.
func (c *Client) Model() string { return c.defaultModel }

// NewSession creates a new session for the Claude API.
// It converts the provided tools to Claude's tool format and initializes a new chat session.
func (c *Client) NewSession(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
	cfg := gollem.NewSessionConfig(options...)

	// Convert gollem.Tool to anthropic.ToolUnionParam
	claudeTools := make([]anthropic.ToolUnionParam, len(cfg.Tools()))
	for i, tool := range cfg.Tools() {
		claudeTools[i] = convertTool(tool)
	}

	issuer := gollem.Issuer{Provider: gollem.LLMTypeClaude, Model: c.defaultModel, Scope: c.issuerScope}

	// Initialize history from config (convert to Claude native format)
	var historyMessages []anthropic.MessageParam
	if cfg.History() != nil {
		var err error
		historyMessages, err = toMessages(cfg.History(), issuer)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert history to Claude format")
		}
	}

	session := &Session{
		apiClient:       &realAPIClient{client: c.client},
		defaultModel:    c.defaultModel,
		tools:           claudeTools,
		params:          c.params,
		historyMessages: historyMessages,
		issuer:          issuer,
		cfg:             cfg,
	}

	return session, nil
}

func (s *Session) History() (*gollem.History, error) {
	return newHistory(s.historyMessages, s.issuer)
}

func (s *Session) AppendHistory(h *gollem.History) error {
	if h == nil {
		return nil
	}
	messages, err := toMessages(h, s.issuer)
	if err != nil {
		return goerr.Wrap(err, "failed to convert history to Claude format")
	}
	s.historyMessages = append(s.historyMessages, messages...)
	return nil
}

// convertInputs converts gollem.Input to Claude messages and tool results
func (s *Session) convertInputs(ctx context.Context, input ...gollem.Input) ([]anthropic.MessageParam, []anthropic.ContentBlockParamUnion, error) {
	return convertGollemInputsToClaude(ctx, input...)
}

// convertGollemInputsToClaude is a shared helper function that converts gollem.Input to Claude messages and tool results
// This function is used by both the standard Claude client and the Vertex AI Claude client to avoid code duplication.
// IMPORTANT: Multiple consecutive Text and Image inputs are combined into a single user message with multiple content blocks,
// as per the Anthropic API specification for multi-modal messages.
func convertGollemInputsToClaude(ctx context.Context, input ...gollem.Input) ([]anthropic.MessageParam, []anthropic.ContentBlockParamUnion, error) {
	var toolResults []anthropic.ContentBlockParamUnion
	var messages []anthropic.MessageParam

	// Accumulate consecutive user content (Text/Image) into a single message
	var userContentBlocks []anthropic.ContentBlockParamUnion

	for _, in := range input {
		switch v := in.(type) {
		case gollem.Text:
			// Skip empty text blocks
			if string(v) == "" {
				continue
			}
			userContentBlocks = append(userContentBlocks, anthropic.NewTextBlock(string(v)))

		case gollem.Image:
			// Create image block for Claude
			imageBlock := anthropic.NewImageBlock(anthropic.Base64ImageSourceParam{
				Type:      "base64",
				MediaType: anthropic.Base64ImageSourceMediaType(v.MimeType()),
				Data:      v.Base64(),
			})
			userContentBlocks = append(userContentBlocks, imageBlock)

		case gollem.PDF:
			// Create document block for Claude using Base64PDFSource
			docBlock := anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{
				Data: v.Base64(),
			})
			userContentBlocks = append(userContentBlocks, docBlock)

		case gollem.FunctionResponse:
			// If we have accumulated user content, create a message for it
			if len(userContentBlocks) > 0 {
				messages = append(messages, anthropic.NewUserMessage(userContentBlocks...))
				userContentBlocks = nil
			}
			// Handle error cases first
			isError := v.Error != nil
			var response string

			if isError {
				response = fmt.Sprintf("Error: %v", v.Error)
			} else {
				data, err := json.Marshal(v.Data)
				if err != nil {
					return nil, nil, goerr.Wrap(err, "failed to marshal function response")
				}
				response = string(data)
			}

			// Create tool result block with new API
			toolResult := anthropic.NewToolResultBlock(v.ID, response, isError)

			// Set content
			if response != "" {
				toolResult.OfToolResult.Content = []anthropic.ToolResultBlockParamContentUnion{
					{OfText: &anthropic.TextBlockParam{Text: response}},
				}
			}

			// Set error flag
			if isError {
				toolResult.OfToolResult.IsError = anthropic.Bool(true)
			}

			toolResults = append(toolResults, toolResult)

		default:
			return nil, nil, goerr.Wrap(gollem.ErrInvalidParameter, "invalid input")
		}
	}

	// Create final user message if there's any remaining user content
	if len(userContentBlocks) > 0 {
		messages = append(messages, anthropic.NewUserMessage(userContentBlocks...))
	}

	if len(toolResults) > 0 {
		messages = append(messages, anthropic.NewUserMessage(toolResults...))
	}

	return messages, toolResults, nil
}

// createSystemPrompt returns the system prompt exactly as the caller configured
// it. It is shared by the Claude API and Vertex AI sessions.
//
// Nothing is appended for a content type or a response schema: Claude accepts
// thinking blocks from earlier turns only while the system prompt is the one
// they were produced with, so text added here per session or per call would
// invalidate them. Without structured outputs the JSON instruction is sent as
// an instruction message instead (see responseFormatInstruction).
func createSystemPrompt(cfg gollem.SessionConfig) []anthropic.TextBlockParam {
	if cfg.SystemPrompt() == "" {
		return nil
	}
	return []anthropic.TextBlockParam{{Text: cfg.SystemPrompt()}}
}

const (
	// jsonInstruction is limited to the next response because the
	// instruction message stays in the history: a later call that asks for
	// text, or for another schema, is sent with every earlier instruction.
	jsonInstruction             = "Format only your next response as valid JSON. This instruction does not apply to any later response."
	jsonSchemaInstructionPrefix = "\n\nYour next response must conform to this JSON Schema:\n"
)

// responseFormatInstruction returns the text of the instruction message for a
// call without structured outputs, or "" when the call does not ask for JSON.
//
// Only the schema in effect is written: the per-call schema if one is given,
// otherwise the session schema when the session content type is JSON. The
// instruction applies to the response that follows it, so naming a second
// schema would ask that one response to conform to two different shapes.
func responseFormatInstruction(cfg gollem.SessionConfig, opts ...gollem.GenerateOption) (string, error) {
	genCfg := gollem.NewGenerateConfig(opts...)
	responseSchema := genCfg.ResponseSchema()
	if responseSchema == nil {
		if cfg.ContentType() != gollem.ContentTypeJSON {
			return "", nil
		}
		responseSchema = cfg.ResponseSchema()
	}

	text := jsonInstruction
	schemaText, err := schema.ConvertParameterToJSONString(responseSchema)
	if err != nil {
		return "", goerr.Wrap(err, "failed to convert response schema to JSON string")
	}
	if schemaText != "" {
		text += jsonSchemaInstructionPrefix + schemaText
	}
	return text, nil
}

// appendInstructionMessage returns turn followed by a user message holding
// only text. turn is not modified.
//
// The instruction is a message of its own rather than a block appended to the
// last input message, so the caller's messages are sent as they were built.
// Claude combines consecutive user messages into one turn, which keeps a
// tool_result ahead of the instruction as the API requires.
func appendInstructionMessage(turn []anthropic.MessageParam, text string) []anthropic.MessageParam {
	result := make([]anthropic.MessageParam, 0, len(turn)+1)
	result = append(result, turn...)
	return append(result, anthropic.NewUserMessage(anthropic.NewTextBlock(text)))
}

// extractJSON extracts JSON from noisy text using jsonex library
// It handles both JSON objects and arrays, with proper error handling and logging
func extractJSON(ctx context.Context, text string) string {
	var jsonResult any
	if err := jsonex.Unmarshal([]byte(text), &jsonResult); err != nil {
		// Not valid JSON or does not contain JSON, return original text
		return text
	}

	jsonBytes, err := json.Marshal(jsonResult)
	if err != nil {
		// Re-marshal failed after successful unmarshal; return original text as fallback
		return text
	}

	return string(jsonBytes)
}

// buildMessageParams builds the Messages request shared by the Claude API and
// Vertex AI sessions: generation parameters, system prompt, tools, response
// schema, per-call overrides and prompt-cache breakpoints, in that order.
//
// history is the conversation so far and turn the messages built from this
// call's inputs. Without structured outputs, a call that asks for JSON gets an
// instruction message appended to turn. The returned turn is the one sent, and
// the caller must store exactly that turn in the history: removing the
// instruction from a later request would edit an earlier user turn, which
// invalidates the prompt cache and the thinking blocks produced after it.
//
// allowStructuredOutputs is false when the client is configured not to send
// output_config.format; structured outputs are then used for no model.
func buildMessageParams(
	model string,
	params generationParameters,
	history, turn []anthropic.MessageParam,
	tools []anthropic.ToolUnionParam,
	cfg gollem.SessionConfig,
	allowStructuredOutputs bool,
	opts ...gollem.GenerateOption,
) (anthropic.MessageNewParams, []anthropic.MessageParam, error) {
	structuredOutputs := allowStructuredOutputs && supportsStructuredOutputs(model)

	if !structuredOutputs {
		instruction, err := responseFormatInstruction(cfg, opts...)
		if err != nil {
			return anthropic.MessageNewParams{}, nil, goerr.Wrap(err, "failed to build response format instruction", goerr.V("model", model))
		}
		if instruction != "" {
			turn = appendInstructionMessage(turn, instruction)
		}
	}

	messages := make([]anthropic.MessageParam, 0, len(history)+len(turn))
	messages = append(messages, history...)
	messages = append(messages, turn...)

	request := anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: params.MaxTokens,
		Messages:  messages,
	}

	// Set temperature and/or top_p (mutually exclusive for Claude)
	if err := setTemperatureAndTopP(&request, params.Temperature, params.TopP); err != nil {
		return anthropic.MessageNewParams{}, nil, goerr.Wrap(err, "failed to set generation parameters")
	}
	if params.effort != "" {
		request.OutputConfig.Effort = anthropic.OutputConfigEffort(params.effort)
	}
	if params.adaptiveThinking {
		request.Thinking = anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{
				Display: anthropic.ThinkingConfigAdaptiveDisplay(params.thinkingDisplay),
			},
		}
	}

	if systemPrompt := createSystemPrompt(cfg); len(systemPrompt) > 0 {
		request.System = systemPrompt
	}

	if len(tools) > 0 {
		request.Tools = tools
	}

	if structuredOutputs {
		if err := applyResponseSchema(&request, cfg, opts...); err != nil {
			return anthropic.MessageNewParams{}, nil, err
		}
	}

	applyPerCallOverrides(&request, opts...)

	// Inject prompt-cache breakpoints on the stable prefix and tail
	if cfg.PromptCache() {
		applyPromptCacheBreakpoints(&request)
	}

	return request, turn, nil
}

type streamBlockKind int

const (
	streamBlockText streamBlockKind = iota
	streamBlockThinking
	streamBlockRedactedThinking
	streamBlockToolUse
)

// streamBlock accumulates one content block of a streamed response.
type streamBlock struct {
	index     int64
	kind      streamBlockKind
	text      strings.Builder
	signature string
	data      string
	toolUse   anthropic.ContentBlockParamUnion
}

// streamBlocks holds the content blocks of a streamed response in the order
// the API started them, so that the history keeps thinking blocks before the
// text and tool use that followed them.
type streamBlocks []*streamBlock

// at returns the block with the given event index, adding one of kind when the
// stream has not started it.
func (x *streamBlocks) at(index int64, kind streamBlockKind) *streamBlock {
	for _, b := range *x {
		if b.index == index {
			return b
		}
	}
	b := &streamBlock{index: index, kind: kind}
	*x = append(*x, b)
	return b
}

// params converts the blocks to the content of an assistant message, keeping
// every block in the order received and its text unchanged. Claude checks the
// signature of a thinking block against the turn it was produced in, so
// joining, reordering or rewriting the blocks of that turn, for example by
// extracting JSON from its text, makes the API reject the thinking blocks
// when the history is sent back.
func (x streamBlocks) params() []anthropic.ContentBlockParamUnion {
	var content []anthropic.ContentBlockParamUnion
	for _, b := range x {
		switch b.kind {
		case streamBlockText:
			// The API rejects an empty text block, and Generate leaves it out of
			// the history as well.
			if b.text.Len() > 0 {
				content = append(content, anthropic.NewTextBlock(b.text.String()))
			}
		case streamBlockThinking:
			content = append(content, anthropic.NewThinkingBlock(b.signature, b.text.String()))
		case streamBlockRedactedThinking:
			content = append(content, anthropic.NewRedactedThinkingBlock(b.data))
		case streamBlockToolUse:
			if b.toolUse.OfToolUse != nil {
				content = append(content, b.toolUse)
			}
		}
	}
	return content
}

// generateClaudeStream reads a Messages API stream and sends its text,
// thinking and tool calls as they arrive. It is shared by the Claude API and
// the Vertex AI Claude clients.
//
// newMessages are the inputs of this call. They are appended to messageHistory
// together with the response only when the stream completes without error, so
// a failed call leaves the history as it was and can be retried as is. The
// response is recorded as received, as Generate records it.
func generateClaudeStream(
	stream *ssestream.Stream[anthropic.MessageStreamEventUnion],
	messageHistory *[]anthropic.MessageParam,
	newMessages []anthropic.MessageParam,
) (<-chan *gollem.ContentResponse, error) {
	if stream == nil {
		return nil, goerr.New("failed to create message stream")
	}

	responseChan := make(chan *gollem.ContentResponse)

	// Accumulate the response blocks, in index order, for message history
	var blocks streamBlocks
	acc := newFunctionCallAccumulator()
	// toolUseStartInput is the input of the latest tool_use content_block_start.
	var toolUseStartInput json.RawMessage
	var totalInputTokens int
	var totalOutputTokens int
	var totalCacheCreation int
	var totalCacheRead int

	go func() {
		defer close(responseChan)
		// Close only releases the response body. By the time it runs, every
		// result, including a stream error, has been sent on responseChan, so a
		// failure to close cannot change what the caller received.
		defer func() { _ = stream.Close() }()

		for {
			if !stream.Next() {
				// An API error such as a 400 ends the stream before any event;
				// without this check the caller would see an empty response.
				if err := stream.Err(); err != nil {
					responseChan <- &gollem.ContentResponse{
						Error: goerr.Wrap(err, "failed to stream message", tokenLimitErrorOptions(err)...),
					}
					return
				}

				*messageHistory = append(*messageHistory, newMessages...)
				// Add accumulated message to history when stream ends
				if content := blocks.params(); len(content) > 0 {
					*messageHistory = append(*messageHistory, anthropic.NewAssistantMessage(content...))
				}
				return
			}

			event := stream.Current()
			response := &gollem.ContentResponse{}

			switch event.Type {
			case "message_delta":
				messageDelta := event.AsMessageDelta()
				if messageDelta.Usage.OutputTokens > 0 {
					totalOutputTokens = int(messageDelta.Usage.OutputTokens)
				}
				if messageDelta.Delta.StopReason != "" {
					response.FinishReason = string(messageDelta.Delta.StopReason)
					response.StopSequence = messageDelta.Delta.StopSequence
					response.Refusal = refusalFrom(messageDelta.Delta.StopReason, messageDelta.Delta.StopDetails)
					response.InputToken = totalInputTokens
					response.OutputToken = totalOutputTokens
					response.CacheCreationInputToken = totalCacheCreation
					response.CacheReadInputToken = totalCacheRead
				}
			case "message_start":
				messageStart := event.AsMessageStart()
				// input_tokens counts only tokens after the last cache breakpoint;
				// restore the cached prefix so InputToken means total input.
				totalCacheCreation = int(messageStart.Message.Usage.CacheCreationInputTokens)
				totalCacheRead = int(messageStart.Message.Usage.CacheReadInputTokens)
				totalInputTokens = int(messageStart.Message.Usage.InputTokens) + totalCacheCreation + totalCacheRead
				if messageStart.Message.Usage.OutputTokens > 0 {
					totalOutputTokens = int(messageStart.Message.Usage.OutputTokens)
				}
			case "content_block_delta":
				deltaEvent := event.AsContentBlockDelta()
				switch deltaEvent.Delta.Type {
				case "text_delta":
					textDelta := deltaEvent.Delta.AsTextDelta()
					response.Texts = append(response.Texts, textDelta.Text)
					response.InputToken = totalInputTokens
					response.OutputToken = totalOutputTokens
					response.CacheCreationInputToken = totalCacheCreation
					response.CacheReadInputToken = totalCacheRead
					blocks.at(deltaEvent.Index, streamBlockText).text.WriteString(textDelta.Text)
				case "thinking_delta":
					thinking := deltaEvent.Delta.AsThinkingDelta().Thinking
					blocks.at(deltaEvent.Index, streamBlockThinking).text.WriteString(thinking)
					if thinking != "" {
						response.Thoughts = append(response.Thoughts, thinking)
						response.InputToken = totalInputTokens
						response.OutputToken = totalOutputTokens
						response.CacheCreationInputToken = totalCacheCreation
						response.CacheReadInputToken = totalCacheRead
					}
				case "signature_delta":
					blocks.at(deltaEvent.Index, streamBlockThinking).signature += deltaEvent.Delta.AsSignatureDelta().Signature
				case "input_json_delta":
					jsonDelta := deltaEvent.Delta.AsInputJSONDelta()
					if jsonDelta.PartialJSON != "" {
						acc.Arguments += jsonDelta.PartialJSON
					}
				}
			case "content_block_start":
				startEvent := event.AsContentBlockStart()
				if t := startEvent.ContentBlock.Type; isOtherBlockType(t) {
					response.OtherBlockTypes = append(response.OtherBlockTypes, t)
				}
				switch startEvent.ContentBlock.Type {
				case "tool_use":
					toolUseBlock := startEvent.ContentBlock.AsToolUse()
					acc.ID = toolUseBlock.ID
					acc.Name = toolUseBlock.Name
					toolUseStartInput = toolUseBlock.Input
					blocks.at(startEvent.Index, streamBlockToolUse)
				case "text":
					blocks.at(startEvent.Index, streamBlockText)
				case "thinking":
					b := blocks.at(startEvent.Index, streamBlockThinking)
					thinking := startEvent.ContentBlock.AsThinking()
					b.text.WriteString(thinking.Thinking)
					b.signature += thinking.Signature
				case "redacted_thinking":
					blocks.at(startEvent.Index, streamBlockRedactedThinking).data = startEvent.ContentBlock.AsRedactedThinking().Data
				}
			case "content_block_stop":
				if acc.ID != "" && acc.Name != "" {
					funcCall, err := acc.accumulate()
					if err != nil {
						response.Error = err
						responseChan <- response
						return
					}
					response.FunctionCalls = append(response.FunctionCalls, funcCall)
					response.InputToken = totalInputTokens
					response.OutputToken = totalOutputTokens
					response.CacheCreationInputToken = totalCacheCreation
					response.CacheReadInputToken = totalCacheRead
					// The session history keeps the arguments JSON as received, as
					// resp.ToParam() does for Generate. accumulate() has already
					// rejected anything that is not a JSON object.
					input := json.RawMessage(acc.Arguments)
					if acc.Arguments == "" {
						input, err = streamedToolUseInputWithoutDelta(toolUseStartInput, funcCall)
						if err != nil {
							response.Error = err
							responseChan <- response
							return
						}
					}
					blocks.at(event.AsContentBlockStop().Index, streamBlockToolUse).toolUse =
						anthropic.NewToolUseBlock(funcCall.ID, input, funcCall.Name)
					acc = newFunctionCallAccumulator()
				}
			}

			if len(response.Texts) > 0 || len(response.Thoughts) > 0 || len(response.FunctionCalls) > 0 ||
				response.FinishReason != "" || len(response.OtherBlockTypes) > 0 {
				responseChan <- response
			}
		}
	}()

	return responseChan, nil
}

// traceClaudeStream passes the responses of ch through and ends the LLM call
// span that the caller started on ctx when the stream ends. newMessages are the
// inputs of this call; previous turns are already captured in earlier spans.
func traceClaudeStream(
	ctx context.Context,
	handler trace.Handler,
	model, systemPrompt string,
	newMessages []anthropic.MessageParam,
	ch <-chan *gollem.ContentResponse,
) <-chan *gollem.ContentResponse {
	wrapped := make(chan *gollem.ContentResponse)
	go func() {
		defer close(wrapped)

		var streamErr error
		var texts, thoughts, otherBlockTypes []string
		var functionCalls []*trace.FunctionCall
		var finishReason string
		var refusal *gollem.Refusal
		var inputTokens, outputTokens, cacheCreation, cacheRead int

		for resp := range ch {
			if resp.Error != nil && streamErr == nil {
				streamErr = resp.Error
			}
			if resp.FinishReason != "" {
				finishReason = resp.FinishReason
			}
			if resp.Refusal != nil {
				refusal = resp.Refusal
			}
			texts = append(texts, resp.Texts...)
			thoughts = append(thoughts, resp.Thoughts...)
			otherBlockTypes = append(otherBlockTypes, resp.OtherBlockTypes...)
			for _, fc := range resp.FunctionCalls {
				functionCalls = append(functionCalls, &trace.FunctionCall{
					ID:        fc.ID,
					Name:      fc.Name,
					Arguments: fc.Arguments,
				})
			}
			if resp.InputToken > 0 {
				inputTokens = resp.InputToken
			}
			if resp.OutputToken > 0 {
				outputTokens = resp.OutputToken
			}
			if resp.CacheCreationInputToken > 0 {
				cacheCreation = resp.CacheCreationInputToken
			}
			if resp.CacheReadInputToken > 0 {
				cacheRead = resp.CacheReadInputToken
			}
			wrapped <- resp
		}

		handler.EndLLMCall(ctx, &trace.LLMCallData{
			InputTokens:              inputTokens,
			OutputTokens:             outputTokens,
			CacheCreationInputTokens: cacheCreation,
			CacheReadInputTokens:     cacheRead,
			Model:                    model,
			Request: &trace.LLMRequest{
				SystemPrompt: systemPrompt,
				Messages:     claudeMessagesToTraceMessages(newMessages),
			},
			Response: &trace.LLMResponse{
				Texts:           texts,
				Thoughts:        thoughts,
				OtherBlockTypes: otherBlockTypes,
				FunctionCalls:   functionCalls,
				FinishReason:    finishReason,
				Refusal:         convert.TraceRefusal(refusal),
			},
		}, streamErr)
	}()
	return wrapped
}

// toResponseStream converts a stream of content responses to the responses a
// Session returns.
func toResponseStream(ch <-chan *gollem.ContentResponse) <-chan *gollem.Response {
	responseChan := make(chan *gollem.Response)
	go func() {
		defer close(responseChan)
		for streamResp := range ch {
			if streamResp.Error != nil {
				responseChan <- &gollem.Response{Error: streamResp.Error}
				continue
			}
			responseChan <- &gollem.Response{
				Texts:                   streamResp.Texts,
				Thoughts:                streamResp.Thoughts,
				FunctionCalls:           streamResp.FunctionCalls,
				InputToken:              streamResp.InputToken,
				OutputToken:             streamResp.OutputToken,
				CacheCreationInputToken: streamResp.CacheCreationInputToken,
				CacheReadInputToken:     streamResp.CacheReadInputToken,
				FinishReason:            streamResp.FinishReason,
				StopSequence:            streamResp.StopSequence,
				OtherBlockTypes:         streamResp.OtherBlockTypes,
				Refusal:                 streamResp.Refusal,
			}
		}
	}()
	return responseChan
}

// refusalFrom returns the refusal details reported with stopReason, or nil
// when the generation did not stop because of a refusal.
func refusalFrom(stopReason anthropic.StopReason, details anthropic.RefusalStopDetails) *gollem.Refusal {
	if stopReason != anthropic.StopReasonRefusal {
		return nil
	}
	refusal := &gollem.Refusal{
		Reason:      string(stopReason),
		Explanation: details.Explanation,
	}
	if details.Category != "" {
		refusal.Categories = []string{string(details.Category)}
	}
	return refusal
}

// isOtherBlockType reports whether a content block of type t is reported in
// Response.OtherBlockTypes.
func isOtherBlockType(t string) bool {
	return t != "text" && t != "tool_use"
}

// appendOtherBlock appends the type of content to types when it is neither
// text nor tool_use, and its thinking text to thoughts when it is a thinking
// block with text. A block whose text was omitted adds no element to thoughts;
// see Response.Thoughts. Both the response and its trace use it, so that they
// report the same values.
func appendOtherBlock(types, thoughts []string, content anthropic.ContentBlockUnion) ([]string, []string) {
	if !isOtherBlockType(content.Type) {
		return types, thoughts
	}
	types = append(types, content.Type)
	if content.Type == "thinking" {
		if thinking := content.AsThinking().Thinking; thinking != "" {
			thoughts = append(thoughts, thinking)
		}
	}
	return types, thoughts
}

// processResponseWithContentType converts Claude response to gollem.Response.
// extractJSONText selects whether JSON is extracted from text blocks; see
// needsJSONExtraction.
//
// A response without content blocks, such as a refusal, still carries the
// usage and the stop reason, so that the caller can tell why no text came back.
func processResponseWithContentType(ctx context.Context, resp *anthropic.Message, extractJSONText bool) *gollem.Response {
	totalInput, cacheCreation, cacheRead := cacheTokensFromUsage(resp.Usage)
	response := &gollem.Response{
		Texts:                   make([]string, 0),
		FunctionCalls:           make([]*gollem.FunctionCall, 0),
		InputToken:              totalInput,
		OutputToken:             int(resp.Usage.OutputTokens),
		CacheCreationInputToken: cacheCreation,
		CacheReadInputToken:     cacheRead,
		FinishReason:            string(resp.StopReason),
		StopSequence:            resp.StopSequence,
		Refusal:                 refusalFrom(resp.StopReason, resp.StopDetails),
	}

	for _, content := range resp.Content {
		response.OtherBlockTypes, response.Thoughts = appendOtherBlock(response.OtherBlockTypes, response.Thoughts, content)

		switch content.Type {
		case "text":
			textBlock := content.AsText()
			text := textBlock.Text

			if extractJSONText {
				text = extractJSON(ctx, text)
			}

			response.Texts = append(response.Texts, text)
		case "tool_use":
			toolUseBlock := content.AsToolUse()
			args, err := jsonutil.DecodeObject(toolUseBlock.Input)
			if err != nil {
				response.Error = goerr.Wrap(err, "failed to unmarshal function arguments")
				return response
			}

			response.FunctionCalls = append(response.FunctionCalls, &gollem.FunctionCall{
				ID:        toolUseBlock.ID,
				Name:      toolUseBlock.Name,
				Arguments: args,
			})
		}
	}

	return response
}

// Generate processes the input and generates a response with optional per-call overrides.
// It handles both text messages and function responses.
func (s *Session) Generate(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
	// Build the content request for middleware
	// Create a copy of the current history to avoid middleware side effects
	var historyCopy *gollem.History
	if len(s.historyMessages) > 0 {
		var err error
		historyCopy, err = newHistory(s.historyMessages, s.issuer)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert history from Claude format")
		}
	}

	contentReq := &gollem.ContentRequest{
		Inputs:       input,
		History:      historyCopy,
		SystemPrompt: s.cfg.SystemPrompt(),
	}

	// Create the base handler that performs the actual API call
	baseHandler := func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
		// Always update history from middleware (even if same address, content may have changed)
		if req.History != nil {
			var err error
			s.historyMessages, err = toMessages(req.History, s.issuer)
			if err != nil {
				return nil, goerr.Wrap(err, "failed to convert history from middleware")
			}
		}

		messages, _, err := s.convertInputs(ctx, req.Inputs...)
		if err != nil {
			return nil, err
		}

		request, messages, err := buildMessageParams(s.defaultModel, s.params, s.historyMessages, messages, s.tools, s.cfg, true, opts...)
		if err != nil {
			return nil, err
		}

		// Start LLM call trace span
		var traceData *trace.LLMCallData
		var llmErr error
		if h := trace.HandlerFrom(ctx); h != nil {
			ctx = h.StartLLMCall(ctx)
			defer func() { h.EndLLMCall(ctx, traceData, llmErr) }()
		}

		resp, err := s.apiClient.MessagesNew(ctx, request)
		if err != nil {
			llmErr = err
			opts := tokenLimitErrorOptions(err)
			return nil, goerr.Wrap(err, "failed to create message", opts...)
		}

		// Process response and extract content
		processedResp := processResponseWithContentType(ctx, resp, needsJSONExtraction(s.cfg, opts...))

		// Set trace data for defer.
		// Record only messages added in this turn; previous turns are already
		// captured in earlier trace spans.
		traceData = buildClaudeTraceData(resp, s.defaultModel, s.cfg.SystemPrompt(), messages)

		// Update history with new messages (already in Claude format)
		s.historyMessages = append(s.historyMessages, messages...)

		// Only add response to history if it has content
		respParam := resp.ToParam()
		if len(respParam.Content) > 0 {
			s.historyMessages = append(s.historyMessages, respParam)
		}

		return &gollem.ContentResponse{
			Texts:                   processedResp.Texts,
			Thoughts:                processedResp.Thoughts,
			FunctionCalls:           processedResp.FunctionCalls,
			InputToken:              processedResp.InputToken,
			OutputToken:             processedResp.OutputToken,
			CacheCreationInputToken: processedResp.CacheCreationInputToken,
			CacheReadInputToken:     processedResp.CacheReadInputToken,
			FinishReason:            processedResp.FinishReason,
			StopSequence:            processedResp.StopSequence,
			OtherBlockTypes:         processedResp.OtherBlockTypes,
			Refusal:                 processedResp.Refusal,
		}, nil
	}

	// Build middleware chain
	handler := gollem.ContentBlockHandler(baseHandler)
	for i := len(s.cfg.ContentBlockMiddlewares()) - 1; i >= 0; i-- {
		handler = s.cfg.ContentBlockMiddlewares()[i](handler)
	}

	// Execute middleware chain
	contentResp, err := handler(ctx, contentReq)
	if err != nil {
		return nil, err
	}

	// Convert ContentResponse back to gollem.Response
	return &gollem.Response{
		Texts:                   contentResp.Texts,
		Thoughts:                contentResp.Thoughts,
		FunctionCalls:           contentResp.FunctionCalls,
		InputToken:              contentResp.InputToken,
		OutputToken:             contentResp.OutputToken,
		CacheCreationInputToken: contentResp.CacheCreationInputToken,
		CacheReadInputToken:     contentResp.CacheReadInputToken,
		FinishReason:            contentResp.FinishReason,
		StopSequence:            contentResp.StopSequence,
		OtherBlockTypes:         contentResp.OtherBlockTypes,
		Refusal:                 contentResp.Refusal,
	}, nil
}

// applyPerCallOverrides applies per-call GenerateOption overrides to Claude
// request params. The per-call response schema is applied by
// applyResponseSchema.
func applyPerCallOverrides(request *anthropic.MessageNewParams, opts ...gollem.GenerateOption) {
	genCfg := gollem.NewGenerateConfig(opts...)
	if m := genCfg.MaxTokens(); m != nil {
		request.MaxTokens = int64(*m)
	}
	// tool_choice is only meaningful with tools; without them there is nothing
	// to forbid, so the field is left unset rather than sent on its own.
	if genCfg.ToolCallsDisabled() && len(request.Tools) > 0 {
		request.ToolChoice = anthropic.ToolChoiceUnionParam{OfNone: &anthropic.ToolChoiceNoneParam{}}
	}
}

// applyResponseSchema sends the response schema in effect for this call as
// output_config.format: the per-call schema if one is given, otherwise the
// session schema when the session content type is JSON. It is used only with
// structured outputs; buildMessageParams sends an instruction message instead
// for a model listed in structuredOutputsUnsupported or a Vertex AI client
// configured with WithVertexStructuredOutputsDisabled.
//
// The system prompt and tool list are not touched: the API accepts thinking
// blocks from earlier turns only while both are unchanged, whereas changing
// output_config does not invalidate them.
func applyResponseSchema(request *anthropic.MessageNewParams, cfg gollem.SessionConfig, opts ...gollem.GenerateOption) error {
	genCfg := gollem.NewGenerateConfig(opts...)
	responseSchema := genCfg.ResponseSchema()
	if responseSchema == nil && cfg.ContentType() == gollem.ContentTypeJSON {
		responseSchema = cfg.ResponseSchema()
	}
	if responseSchema == nil {
		return nil
	}
	format, err := outputFormat(responseSchema)
	if err != nil {
		return goerr.Wrap(err, "failed to convert response schema", goerr.V("model", request.Model))
	}
	request.OutputConfig.Format = format
	return nil
}

// needsJSONExtraction reports whether JSON is extracted from the response text:
// whenever the content type in effect for this call is JSON, however the
// schema was sent.
func needsJSONExtraction(cfg gollem.SessionConfig, opts ...gollem.GenerateOption) bool {
	return effectiveContentType(cfg.ContentType(), opts...) == gollem.ContentTypeJSON
}

// applyPromptCacheBreakpoints marks the stable prefix (system prompt, tools) and
// the growing conversation tail with ephemeral cache_control so Claude serves
// repeated prefixes from its prompt cache. Empty sections are skipped. It never
// mutates shared session/history state: it copies the single slice element it
// marks. Content below a model's minimum cacheable length is a no-op on the API
// side (no error is returned), so no token-count guard is needed here.
func applyPromptCacheBreakpoints(request *anthropic.MessageNewParams) {
	// TTL is set explicitly to the default 5m. A zero-value
	// CacheControlEphemeralParam{} is dropped by the SDK's `omitzero` encoding
	// (it has no IsZero override and reflects as zero), which would silently emit
	// no cache_control at all; setting TTL forces the field to be present.
	cc := anthropic.CacheControlEphemeralParam{TTL: anthropic.CacheControlEphemeralTTLTTL5m}

	// System: last block. createSystemPrompt returns a fresh slice each call,
	// so in-place assignment does not touch shared state.
	if n := len(request.System); n > 0 {
		request.System[n-1].CacheControl = cc
	}

	// Tools: last tool. request.Tools aliases the session's tool slice, so copy
	// the header and the target ToolParam before marking.
	if n := len(request.Tools); n > 0 {
		if src := request.Tools[n-1].OfTool; src != nil {
			tools := make([]anthropic.ToolUnionParam, n)
			copy(tools, request.Tools)
			toolCopy := *src
			toolCopy.CacheControl = cc
			tools[n-1].OfTool = &toolCopy
			request.Tools = tools
		}
	}

	// Conversation tail: last content block of the last message.
	markMessageTail(request.Messages, cc)
}

// markMessageTail marks the last content block of the last message with cc,
// copying the message's content slice and the target block so shared native
// history is never mutated. msgs must be a freshly built slice; only its last
// element is reassigned.
func markMessageTail(msgs []anthropic.MessageParam, cc anthropic.CacheControlEphemeralParam) {
	if len(msgs) == 0 {
		return
	}
	last := msgs[len(msgs)-1]
	if len(last.Content) == 0 {
		return
	}
	content := make([]anthropic.ContentBlockParamUnion, len(last.Content))
	copy(content, last.Content)
	m := len(content) - 1
	switch {
	case content[m].OfText != nil:
		b := *content[m].OfText
		b.CacheControl = cc
		content[m].OfText = &b
	case content[m].OfImage != nil:
		b := *content[m].OfImage
		b.CacheControl = cc
		content[m].OfImage = &b
	case content[m].OfDocument != nil:
		b := *content[m].OfDocument
		b.CacheControl = cc
		content[m].OfDocument = &b
	case content[m].OfToolResult != nil:
		b := *content[m].OfToolResult
		b.CacheControl = cc
		content[m].OfToolResult = &b
	default:
		return // unknown variant: skip rather than guess
	}
	last.Content = content
	msgs[len(msgs)-1] = last
}

// cacheTokensFromUsage extracts prompt-cache token counts from a Claude usage
// object. Anthropic's input_tokens reports only the tokens after the last cache
// breakpoint, so the returned total input restores the cached prefix (creation +
// read) to keep gollem's InputToken meaning "total input" for existing consumers
// (e.g. the compacter). When caching did not occur, creation and read are 0 and
// total equals the raw input_tokens.
func cacheTokensFromUsage(u anthropic.Usage) (totalInput, creation, read int) {
	creation = int(u.CacheCreationInputTokens)
	read = int(u.CacheReadInputTokens)
	totalInput = int(u.InputTokens) + creation + read
	return
}

// effectiveContentType returns the content type considering per-call schema override.
func effectiveContentType(sessionContentType gollem.ContentType, opts ...gollem.GenerateOption) gollem.ContentType {
	genCfg := gollem.NewGenerateConfig(opts...)
	if genCfg.ResponseSchema() != nil {
		return gollem.ContentTypeJSON
	}
	return sessionContentType
}

// FunctionCallAccumulator accumulates function call information from stream
type FunctionCallAccumulator struct {
	ID        string
	Name      string
	Arguments string
}

func newFunctionCallAccumulator() *FunctionCallAccumulator {
	return &FunctionCallAccumulator{
		Arguments: "",
	}
}

func (a *FunctionCallAccumulator) accumulate() (*gollem.FunctionCall, error) {
	if a.ID == "" || a.Name == "" {
		return nil, goerr.Wrap(gollem.ErrInvalidParameter, "function call is not complete")
	}

	var args map[string]any
	if a.Arguments != "" {
		decoded, err := jsonutil.DecodeObject([]byte(a.Arguments))
		if err != nil {
			return nil, goerr.Wrap(err, "failed to unmarshal function call arguments", goerr.V("accumulator", a))
		}
		args = decoded
	}

	return &gollem.FunctionCall{
		ID:        a.ID,
		Name:      a.Name,
		Arguments: args,
	}, nil
}

// streamedToolUseInputWithoutDelta returns the input of a streamed tool_use block
// for which no input_json_delta arrived. The input of content_block_start, {} for
// a call without arguments, is then the whole input the API returned. When it is
// not a JSON object, the arguments are encoded from funcCall as before.
func streamedToolUseInputWithoutDelta(startInput json.RawMessage, funcCall *gollem.FunctionCall) (json.RawMessage, error) {
	if len(startInput) > 0 {
		args, err := jsonutil.DecodeObject(startInput)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to decode tool call input", goerr.V("tool", funcCall.Name))
		}
		if args != nil {
			return startInput, nil
		}
	}
	input, err := toolUseInput(funcCall.Arguments)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to encode tool call arguments", goerr.V("tool", funcCall.Name))
	}
	return input, nil
}

// Stream processes the input and generates a response stream with optional per-call overrides.
// It handles both text messages and function responses, and returns a channel for streaming responses.
func (s *Session) Stream(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (<-chan *gollem.Response, error) {
	// Build the content request for middleware
	// Create a copy of the current history to avoid middleware side effects
	var historyCopy *gollem.History
	if len(s.historyMessages) > 0 {
		var err error
		historyCopy, err = newHistory(s.historyMessages, s.issuer)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert history from Claude format")
		}
	}

	contentReq := &gollem.ContentRequest{
		Inputs:       input,
		History:      historyCopy,
		SystemPrompt: s.cfg.SystemPrompt(),
	}

	// Create the base handler that performs the actual API call
	baseHandler := func(ctx context.Context, req *gollem.ContentRequest) (<-chan *gollem.ContentResponse, error) {
		// Update history if modified by middleware
		if req.History != nil {
			var err error
			s.historyMessages, err = toMessages(req.History, s.issuer)
			if err != nil {
				return nil, goerr.Wrap(err, "failed to convert history from middleware")
			}
		}

		messages, _, err := s.convertInputs(ctx, req.Inputs...)
		if err != nil {
			return nil, err
		}

		request, messages, err := buildMessageParams(s.defaultModel, s.params, s.historyMessages, messages, s.tools, s.cfg, true, opts...)
		if err != nil {
			return nil, err
		}

		traceHandler := trace.HandlerFrom(ctx)
		if traceHandler != nil {
			ctx = traceHandler.StartLLMCall(ctx)
		}

		ch, err := generateClaudeStream(s.apiClient.MessagesNewStreaming(ctx, request), &s.historyMessages, messages)
		if err != nil {
			if traceHandler != nil {
				traceHandler.EndLLMCall(ctx, nil, err)
			}
			return nil, err
		}
		if traceHandler == nil {
			return ch, nil
		}
		return traceClaudeStream(ctx, traceHandler, s.defaultModel, s.cfg.SystemPrompt(), messages, ch), nil
	}

	// Build middleware chain
	handler := gollem.ContentStreamHandler(baseHandler)
	for i := len(s.cfg.ContentStreamMiddlewares()) - 1; i >= 0; i-- {
		handler = s.cfg.ContentStreamMiddlewares()[i](handler)
	}

	// Execute middleware chain
	streamChan, err := handler(ctx, contentReq)
	if err != nil {
		return nil, err
	}

	return toResponseStream(streamChan), nil
}

// countTokensWithParams is a helper function that builds the count tokens parameters
// and calls the API.
func countTokensWithParams(
	ctx context.Context,
	model string,
	historyMessages []anthropic.MessageParam,
	newMessages []anthropic.MessageParam,
	systemPrompt string,
	tools []anthropic.ToolUnionParam,
	apiClient apiClient,
) (int, error) {
	// Build complete messages list: history + new inputs
	apiMessages := make([]anthropic.MessageParam, 0, len(historyMessages)+len(newMessages))
	apiMessages = append(apiMessages, historyMessages...)
	apiMessages = append(apiMessages, newMessages...)

	// Prepare count tokens parameters
	params := anthropic.MessageCountTokensParams{
		Model:    anthropic.Model(model),
		Messages: apiMessages,
	}

	// Add system prompt if available
	if systemPrompt != "" {
		params.System = anthropic.MessageCountTokensParamsSystemUnion{
			OfString: anthropic.String(systemPrompt),
		}
	}

	// Add tools if available
	if len(tools) > 0 {
		// Convert ToolUnionParam to MessageCountTokensToolUnionParam
		countTools := make([]anthropic.MessageCountTokensToolUnionParam, 0, len(tools))
		for _, tool := range tools {
			if tool.OfTool != nil {
				countTools = append(countTools, anthropic.MessageCountTokensToolUnionParam{
					OfTool: tool.OfTool,
				})
			}
		}
		params.Tools = countTools
	}

	// Start LLM call trace span
	var traceData *trace.LLMCallData
	var llmErr error
	if h := trace.HandlerFrom(ctx); h != nil {
		ctx = h.StartLLMCall(ctx)
		defer func() { h.EndLLMCall(ctx, traceData, llmErr) }()
	}

	// Call the CountTokens API
	result, err := apiClient.MessagesCountTokens(ctx, params)
	if err != nil {
		llmErr = err
		return 0, goerr.Wrap(err, "failed to count tokens")
	}

	traceData = &trace.LLMCallData{
		InputTokens: int(result.InputTokens),
		Model:       string(params.Model),
		Request: &trace.LLMRequest{
			SystemPrompt: systemPrompt,
		},
		Response: &trace.LLMResponse{},
	}

	return int(result.InputTokens), nil
}

// CountToken calculates the total number of tokens for the given inputs,
// including system prompt, history messages, and new inputs.
// This uses Anthropic's Messages Count Tokens API.
func (s *Session) CountToken(ctx context.Context, input ...gollem.Input) (int, error) {
	// Convert inputs to Claude messages
	messages, _, err := s.convertInputs(ctx, input...)
	if err != nil {
		return 0, goerr.Wrap(err, "failed to convert inputs for token counting")
	}

	// Create copies of historyMessages and tools to avoid race conditions
	// This ensures thread safety when reading session state
	historyMessagesCopy := make([]anthropic.MessageParam, len(s.historyMessages))
	copy(historyMessagesCopy, s.historyMessages)

	toolsCopy := make([]anthropic.ToolUnionParam, len(s.tools))
	copy(toolsCopy, s.tools)

	return countTokensWithParams(
		ctx,
		s.defaultModel,
		historyMessagesCopy,
		messages,
		s.cfg.SystemPrompt(),
		toolsCopy,
		s.apiClient,
	)
}

// tokenLimitErrorOptions checks if the error is a token limit exceeded error
// and returns goerr.Option to tag the error with ErrTagTokenExceeded.
// Returns nil if the error is not a token limit exceeded error.
//
// Detection logic:
// - Error must be *anthropic.Error
// - StatusCode must be 400 or 413 (Request Entity Too Large)
// - Parse RawJSON() to get error structure
// - error.type must be "invalid_request_error"
// - error.message must contain "prompt is too long" (case-insensitive)
func tokenLimitErrorOptions(err error) []goerr.Option {
	var apiErr *anthropic.Error
	if !errors.As(err, &apiErr) {
		return nil
	}

	if apiErr.StatusCode != 400 && apiErr.StatusCode != 413 {
		return nil
	}

	// Parse RawJSON to get error details
	type claudeErrorWrapper struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}

	var wrapper claudeErrorWrapper
	if err := json.Unmarshal([]byte(apiErr.RawJSON()), &wrapper); err != nil {
		return nil
	}

	if wrapper.Error.Type != "invalid_request_error" {
		return nil
	}

	// Check for token limit error message (case-insensitive)
	lowerMessage := strings.ToLower(wrapper.Error.Message)
	if strings.Contains(lowerMessage, "prompt is too long") {
		return []goerr.Option{goerr.Tag(gollem.ErrTagTokenExceeded)}
	}

	return nil
}

// claudeMessagesToTraceMessages converts Claude message params to trace messages.
func claudeMessagesToTraceMessages(messages []anthropic.MessageParam) []trace.Message {
	var result []trace.Message
	for _, msg := range messages {
		var blocks []trace.MessageContent
		for _, block := range msg.Content {
			switch {
			case block.OfText != nil:
				blocks = append(blocks, trace.NewTextContent(block.OfText.Text))
			case block.OfToolUse != nil:
				// The input is a json.RawMessage for every block this package builds (see
				// toolUseInput); a map only reaches here from a caller that assembled the
				// message itself. Trace data is diagnostic, so a value that fits neither
				// shape is recorded as no arguments rather than failing the request.
				var args map[string]any
				switch input := block.OfToolUse.Input.(type) {
				case json.RawMessage:
					if decoded, err := jsonutil.DecodeObject(input); err == nil {
						args = decoded
					}
				case map[string]any:
					args = input
				}
				blocks = append(blocks, trace.NewToolCallContent(
					block.OfToolUse.ID, block.OfToolUse.Name, args,
				))
			case block.OfToolResult != nil:
				blocks = append(blocks, trace.NewToolResponseContent(
					block.OfToolResult.ToolUseID, "", nil,
				))
				for _, c := range block.OfToolResult.Content {
					switch {
					case c.OfText != nil:
						blocks = append(blocks, trace.NewTextContent(c.OfText.Text))
					case c.OfImage != nil:
						mc := trace.NewMediaContent("image", "")
						if mt := c.OfImage.Source.GetMediaType(); mt != nil {
							mc.MediaType = *mt
						}
						blocks = append(blocks, mc)
					}
				}
			case block.OfImage != nil:
				mc := trace.NewMediaContent("image", "")
				if mt := block.OfImage.Source.GetMediaType(); mt != nil {
					mc.MediaType = *mt
				}
				if block.OfImage.Source.OfURL != nil {
					mc.URL = block.OfImage.Source.OfURL.URL
				}
				blocks = append(blocks, mc)
			case block.OfDocument != nil:
				mc := trace.NewMediaContent("document", "")
				if mt := block.OfDocument.Source.GetMediaType(); mt != nil {
					mc.MediaType = *mt
				}
				if block.OfDocument.Source.OfURL != nil {
					mc.URL = block.OfDocument.Source.OfURL.URL
				}
				if block.OfDocument.Title.Valid() {
					mc.Title = block.OfDocument.Title.Value
				}
				blocks = append(blocks, mc)
			case block.OfThinking != nil:
				blocks = append(blocks, trace.NewThinkingContent(block.OfThinking.Thinking))
			case block.OfRedactedThinking != nil:
				blocks = append(blocks, trace.NewRedactedThinkingContent())
			}
		}
		if len(blocks) > 0 {
			result = append(result, trace.Message{
				Role:     string(msg.Role),
				Contents: blocks,
			})
		}
	}
	return result
}

// buildClaudeTraceData builds trace.LLMCallData from a Claude API response.
func buildClaudeTraceData(resp *anthropic.Message, model string, systemPrompt string, messages []anthropic.MessageParam) *trace.LLMCallData {
	totalInput, cacheCreation, cacheRead := cacheTokensFromUsage(resp.Usage)
	data := &trace.LLMCallData{
		InputTokens:              totalInput,
		OutputTokens:             int(resp.Usage.OutputTokens),
		Model:                    string(resp.Model),
		CacheCreationInputTokens: cacheCreation,
		CacheReadInputTokens:     cacheRead,
		Request: &trace.LLMRequest{
			SystemPrompt: systemPrompt,
			Messages:     claudeMessagesToTraceMessages(messages),
		},
		Response: &trace.LLMResponse{
			FinishReason: string(resp.StopReason),
			Refusal:      convert.TraceRefusal(refusalFrom(resp.StopReason, resp.StopDetails)),
		},
	}

	for _, content := range resp.Content {
		data.Response.OtherBlockTypes, data.Response.Thoughts = appendOtherBlock(data.Response.OtherBlockTypes, data.Response.Thoughts, content)

		switch content.Type {
		case "text":
			data.Response.Texts = append(data.Response.Texts, content.AsText().Text)
		case "tool_use":
			toolUse := content.AsToolUse()
			var args map[string]any
			if err := json.Unmarshal(toolUse.Input, &args); err != nil {
				args = map[string]any{
					"__raw_arguments": string(toolUse.Input),
					"__error":         err.Error(),
				}
			}
			data.Response.FunctionCalls = append(data.Response.FunctionCalls, &trace.FunctionCall{
				ID:        toolUse.ID,
				Name:      toolUse.Name,
				Arguments: args,
			})
		}
	}

	return data
}
