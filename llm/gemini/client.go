package gemini

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/internal/convert"
	gollemschema "github.com/gollem-dev/gollem/internal/schema"
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/goerr/v2"
	"google.golang.org/api/option"
	"google.golang.org/genai"
)

const (
	// DefaultModel is the Gemini model used when no WithModel option is given.
	DefaultModel          = "gemini-3.8-flash"
	DefaultEmbeddingModel = "text-embedding-004"
)

// Client is a client for the Gemini API.
// It provides methods to interact with Google's Gemini models.
type Client struct {
	projectID string
	location  string

	// client is the underlying Gemini client.
	client *genai.Client

	// defaultModel is the model to use for chat completions.
	// It can be overridden using WithModel option.
	defaultModel string

	// embeddingModel is the model to use for embeddings.
	// It can be overridden using WithEmbeddingModel option.
	embeddingModel string

	// gcpOptions are additional options for Google Cloud Platform.
	// They can be set using WithGoogleCloudOptions.
	gcpOptions []option.ClientOption

	// generationConfig contains the default generation parameters
	generationConfig *genai.GenerateContentConfig

	// systemPrompt is the system prompt to use for chat completions.
	systemPrompt string

	// contentType is the type of content to be generated.
	contentType gollem.ContentType

	// issuerScope distinguishes this client's provider-bound data from data
	// issued by another client of the same model. See WithIssuerScope.
	issuerScope string
}

// Option is a configuration option for the Gemini client.
type Option func(*Client)

// WithModel sets the model to use for text generation.
// Default: "gemini-3.8-flash"
func WithModel(model string) Option {
	return func(c *Client) {
		c.defaultModel = model
	}
}

// WithEmbeddingModel sets the model to use for embeddings.
// Default: "text-embedding-004"
func WithEmbeddingModel(model string) Option {
	return func(c *Client) {
		c.embeddingModel = model
	}
}

// WithGoogleCloudOptions sets additional Google Cloud options.
// These can include authentication credentials, endpoint overrides, etc.
func WithGoogleCloudOptions(opts ...option.ClientOption) Option {
	return func(c *Client) {
		c.gcpOptions = append(c.gcpOptions, opts...)
	}
}

// WithTemperature sets the temperature parameter for text generation.
// Controls randomness in output generation.
// Range: 0.0 to 2.0
// Default: 1.0
func WithTemperature(temp float32) Option {
	return func(c *Client) {
		if c.generationConfig == nil {
			c.generationConfig = &genai.GenerateContentConfig{}
		}
		c.generationConfig.Temperature = &temp
	}
}

// WithTopP sets the top_p parameter for text generation.
// Controls diversity via nucleus sampling.
// Range: 0.0 to 1.0
// Default: 1.0
func WithTopP(topP float32) Option {
	return func(c *Client) {
		if c.generationConfig == nil {
			c.generationConfig = &genai.GenerateContentConfig{}
		}
		c.generationConfig.TopP = &topP
	}
}

// WithTopK sets the top_k parameter for text generation.
// Controls diversity via top-k sampling.
// Range: 1 to 40
func WithTopK(topK float32) Option {
	return func(c *Client) {
		if c.generationConfig == nil {
			c.generationConfig = &genai.GenerateContentConfig{}
		}
		topKFloat32 := topK
		c.generationConfig.TopK = &topKFloat32
	}
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(maxTokens int32) Option {
	return func(c *Client) {
		if c.generationConfig == nil {
			c.generationConfig = &genai.GenerateContentConfig{}
		}
		c.generationConfig.MaxOutputTokens = maxTokens
	}
}

// WithStopSequences sets the stop sequences for text generation.
func WithStopSequences(stopSequences []string) Option {
	return func(c *Client) {
		if c.generationConfig == nil {
			c.generationConfig = &genai.GenerateContentConfig{}
		}
		c.generationConfig.StopSequences = stopSequences
	}
}

// WithThinkingBudget sets the thinking budget for text generation.
// A value of -1 enables automatic thinking budget allocation.
//
// Gemini 3.x deprecates thinking_budget in favor of thinking_level; prefer
// WithThinkingLevel for those models. The two options are mutually exclusive
// at the API layer, so calling this clears any thinking level previously set.
func WithThinkingBudget(budget int32) Option {
	return func(c *Client) {
		if c.generationConfig == nil {
			c.generationConfig = &genai.GenerateContentConfig{}
		}
		if c.generationConfig.ThinkingConfig == nil {
			c.generationConfig.ThinkingConfig = &genai.ThinkingConfig{}
		}
		c.generationConfig.ThinkingConfig.ThinkingBudget = &budget
		c.generationConfig.ThinkingConfig.ThinkingLevel = ""
	}
}

// ThinkingLevel is the amount of reasoning a Gemini model performs before it
// answers. It is passed to WithThinkingLevel.
type ThinkingLevel string

// Thinking levels accepted by Gemini 3.x. Each value is sent to the API as is.
const (
	ThinkingLevelMinimal ThinkingLevel = "MINIMAL"
	ThinkingLevelLow     ThinkingLevel = "LOW"
	ThinkingLevelMedium  ThinkingLevel = "MEDIUM"
	ThinkingLevelHigh    ThinkingLevel = "HIGH"
)

// WithThinkingLevel sets the thinking level for text generation.
// Introduced in Gemini 3.x as the replacement for WithThinkingBudget.
//
// Valid values: ThinkingLevelMinimal, ThinkingLevelLow, ThinkingLevelMedium,
// ThinkingLevelHigh. Not every model accepts every level (gemini-3.7-flash and
// gemini-3.8-flash reject ThinkingLevelMinimal); without this option the
// model's own default applies.
//
// Vertex AI rejects requests that carry both thinking_budget and thinking_level
// (HTTP 400), so calling this clears any thinking budget previously set.
func WithThinkingLevel(level ThinkingLevel) Option {
	return func(c *Client) {
		if c.generationConfig == nil {
			c.generationConfig = &genai.GenerateContentConfig{}
		}
		if c.generationConfig.ThinkingConfig == nil {
			c.generationConfig.ThinkingConfig = &genai.ThinkingConfig{}
		}
		// The level is converted rather than mapped through a table so that a level
		// the API adds later can be sent without a gollem release; an Option cannot
		// return an error, so a table would have to drop unknown levels silently.
		c.generationConfig.ThinkingConfig.ThinkingLevel = genai.ThinkingLevel(level)
		c.generationConfig.ThinkingConfig.ThinkingBudget = nil
	}
}

// WithIncludeThoughts requests thought summaries in the response.
// Gemini returns thought parts only when this is enabled, so Response.Thoughts
// stays empty without it. Default: disabled.
//
// This controls the reasoning text only. The thought signatures that Gemini 3.x
// requires for multi-turn tool use are returned regardless of this setting.
func WithIncludeThoughts(include bool) Option {
	return func(c *Client) {
		if c.generationConfig == nil {
			c.generationConfig = &genai.GenerateContentConfig{}
		}
		if c.generationConfig.ThinkingConfig == nil {
			c.generationConfig.ThinkingConfig = &genai.ThinkingConfig{}
		}
		c.generationConfig.ThinkingConfig.IncludeThoughts = include
	}
}

// WithSystemPrompt sets the system prompt to use for chat completions.
func WithSystemPrompt(prompt string) Option {
	return func(c *Client) {
		c.systemPrompt = prompt
	}
}

// WithContentType sets the content type for text generation.
// This determines the format of the generated content.
func WithContentType(contentType gollem.ContentType) Option {
	return func(c *Client) {
		c.contentType = contentType
	}
}

// WithIssuerScope sets the scope recorded on the provider-bound data this
// client creates, such as thought signatures. Data is sent back only to a
// session whose client has the same model and the same scope. Set a distinct
// scope when clients of the same model must not exchange that data, for
// example clients of different Google Cloud projects.
// Default: "" (empty).
func WithIssuerScope(scope string) Option {
	return func(c *Client) {
		c.issuerScope = scope
	}
}

// newConfiguredClient builds a Client from the defaults and the given options,
// stopping short of the parts that need GCP credentials. It is split out of New
// so that tests can exercise the real defaults and option handling without
// reaching Vertex AI.
func newConfiguredClient(projectID, location string, options ...Option) *Client {
	client := &Client{
		projectID:        projectID,
		location:         location,
		defaultModel:     DefaultModel,
		embeddingModel:   DefaultEmbeddingModel,
		contentType:      gollem.ContentTypeText,
		generationConfig: &genai.GenerateContentConfig{},
	}

	for _, option := range options {
		option(client)
	}

	return client
}

// New creates a new client for the Gemini API.
// It requires a project ID and location, and can be configured with additional options.
//
// No thinking configuration is sent unless WithThinkingLevel or
// WithThinkingBudget is given, so each model applies its own default. Sending a
// fixed level instead would break models that accept only a subset of the
// levels (gemini-3-pro-preview takes LOW and HIGH) and models that do not
// accept thinking_level at all (Gemini 2.x).
func New(ctx context.Context, projectID, location string, options ...Option) (*Client, error) {
	if projectID == "" {
		return nil, goerr.New("projectID is required")
	}
	if location == "" {
		return nil, goerr.New("location is required")
	}

	client := newConfiguredClient(projectID, location, options...)

	// Create client configuration for Vertex AI backend
	config := &genai.ClientConfig{
		Project:  projectID,
		Location: location,
		Backend:  genai.BackendVertexAI,
	}

	genaiClient, err := genai.NewClient(ctx, config)
	if err != nil {
		return nil, err
	}

	client.client = genaiClient
	return client, nil
}

// Model returns the model name this client generates through. It is the name
// the client was configured with, so a caller can key its own tables by the
// same string it passed to WithModel.
func (c *Client) Model() string { return c.defaultModel }

// NewSession creates a new session for the Gemini API.
// It converts the provided tools to Gemini's tool format and initializes a new chat session.
func (c *Client) NewSession(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
	cfg := gollem.NewSessionConfig(options...)

	// Prepare generation config
	config := &genai.GenerateContentConfig{}

	// Copy generation config from client
	if c.generationConfig != nil {
		*config = *c.generationConfig
	}

	// Override with session-specific content type
	switch cfg.ContentType() {
	case gollem.ContentTypeJSON:
		config.ResponseMIMEType = "application/json"
	case gollem.ContentTypeText:
		config.ResponseMIMEType = "text/plain"
	}

	// Set response schema if provided
	if cfg.ResponseSchema() != nil {
		if err := setResponseSchema(config, cfg.ResponseSchema()); err != nil {
			return nil, goerr.Wrap(err, "failed to convert response schema")
		}
	}

	// Set system prompt
	systemPrompt := cfg.SystemPrompt()
	if systemPrompt == "" {
		systemPrompt = c.systemPrompt
	}
	if systemPrompt != "" {
		config.SystemInstruction = &genai.Content{
			Role: "system",
			Parts: []*genai.Part{
				{Text: systemPrompt},
			},
		}
	}

	// Convert tools
	if len(cfg.Tools()) > 0 {
		tools := make([]*genai.Tool, 1)
		tools[0] = &genai.Tool{
			FunctionDeclarations: make([]*genai.FunctionDeclaration, len(cfg.Tools())),
		}
		for i, tool := range cfg.Tools() {
			tools[0].FunctionDeclarations[i] = convertToolToNewSDK(tool)
		}
		config.Tools = tools
	}

	issuer := gollem.Issuer{Provider: gollem.LLMTypeGemini, Model: c.defaultModel, Scope: c.issuerScope}

	// Initialize history from config (convert to Gemini native format)
	var historyContents []*genai.Content
	if cfg.History() != nil {
		var err error
		historyContents, err = toContents(cfg.History(), issuer)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert history to Gemini format")
		}
	}

	session := &Session{
		apiClient:       &realAPIClient{client: c.client},
		model:           c.defaultModel,
		config:          config,
		historyContents: historyContents,
		issuer:          issuer,
		cfg:             cfg,
	}

	return session, nil
}

// Session is a session for the Gemini chat.
// It maintains the conversation state and handles message generation.
type Session struct {
	// apiClient is the API client interface for dependency injection
	apiClient apiClient

	// model is the model name to use
	model string

	// config is the generation configuration
	config *genai.GenerateContentConfig

	// historyContents maintains history in Gemini native format for efficiency
	historyContents []*genai.Content

	// issuer identifies this session as the issuer of provider-bound data.
	issuer gollem.Issuer

	// cfg is the session configuration
	cfg gollem.SessionConfig
}

func (s *Session) History() (*gollem.History, error) {
	return newHistory(s.historyContents, s.issuer)
}

func (s *Session) AppendHistory(h *gollem.History) error {
	if h == nil {
		return nil
	}
	contents, err := toContents(h, s.issuer)
	if err != nil {
		return goerr.Wrap(err, "failed to convert history to Gemini format")
	}
	s.historyContents = append(s.historyContents, contents...)
	return nil
}

// convertInputs converts gollem.Input to Gemini parts
func (s *Session) convertInputs(input ...gollem.Input) ([]*genai.Part, error) {
	parts := make([]*genai.Part, 0, len(input))

	for _, in := range input {
		switch v := in.(type) {
		case gollem.Text:
			parts = append(parts, &genai.Part{Text: string(v)})
		case gollem.Image:
			// Check if format is supported by Gemini (no GIF support)
			if v.MimeType() == string(gollem.ImageMimeTypeGIF) {
				return nil, goerr.New("GIF format is not supported by Gemini", goerr.V("mime_type", v.MimeType()))
			}

			parts = append(parts, &genai.Part{
				InlineData: &genai.Blob{
					MIMEType: v.MimeType(),
					Data:     v.Data(),
				},
			})
		case gollem.PDF:
			parts = append(parts, &genai.Part{
				InlineData: &genai.Blob{
					MIMEType: "application/pdf",
					Data:     v.Data(),
				},
			})
		case gollem.FunctionResponse:
			// Propagate the FunctionResponse id so Gemini 3.x can match it back
			// to the corresponding FunctionCall. If the id is a gollem-internal
			// fallback, strip it — feeding Gemini a fabricated id would break
			// strict matching just as badly as no id at all.
			id := v.ID
			if isGeminiFallbackToolCallID(id) {
				id = ""
			}
			if v.Error != nil {
				parts = append(parts, &genai.Part{
					FunctionResponse: &genai.FunctionResponse{
						ID:   id,
						Name: v.Name,
						Response: map[string]any{
							"error_message": fmt.Sprintf("%+v", v.Error),
						},
					},
				})
			} else {
				parts = append(parts, &genai.Part{
					FunctionResponse: &genai.FunctionResponse{
						ID:       id,
						Name:     v.Name,
						Response: v.Data,
					},
				})
			}
		default:
			return nil, goerr.Wrap(gollem.ErrInvalidParameter, "invalid input")
		}
	}
	return parts, nil
}

// processResponse converts Gemini response to gollem.Response. It returns an
// error for a finish reason that gollem reports as an error.
func processResponse(resp *genai.GenerateContentResponse) (*gollem.Response, error) {
	if err := finishReasonError(resp); err != nil {
		return nil, err
	}
	return convertResponse(resp), nil
}

// finishReasonError returns the error for a candidate whose finish reason
// gollem reports as an error, or nil when there is none.
func finishReasonError(resp *genai.GenerateContentResponse) error {
	for _, candidate := range resp.Candidates {
		if strings.Contains(string(candidate.FinishReason), "MALFORMED_FUNCTION_CALL") {
			return goerr.Wrap(gollem.ErrFunctionCallFormat, "malformed function call")
		}
		if strings.Contains(string(candidate.FinishReason), "PROHIBITED_CONTENT") {
			return goerr.Wrap(gollem.ErrProhibitedContent, "prohibited content",
				goerr.V("refusal_reason", string(candidate.FinishReason)),
				goerr.V("refusal_categories", blockedCategories(candidate.SafetyRatings)),
				goerr.V("refusal_explanation", candidate.FinishMessage),
			)
		}
	}
	return nil
}

// convertResponse converts Gemini response to gollem.Response regardless of
// the finish reason. Besides processResponse, it builds the trace record of a
// response that processResponse rejected.
func convertResponse(resp *genai.GenerateContentResponse) *gollem.Response {
	// A blocked prompt has no candidates; the block is reported only in
	// promptFeedback.
	if len(resp.Candidates) == 0 {
		return &gollem.Response{Refusal: promptRefusal(resp.PromptFeedback)}
	}

	response := &gollem.Response{
		Texts:         make([]string, 0),
		FunctionCalls: make([]*gollem.FunctionCall, 0),
		Thoughts:      make([]string, 0),
	}

	// Extract token counts from UsageMetadata if available.
	// PromptTokenCount already includes cached-content tokens, so InputToken
	// stays total; the cached portion is surfaced for observability. Gemini
	// implicit caching does not report cache writes (creation stays 0).
	// OutputToken adds ThoughtsTokenCount to CandidatesTokenCount because Gemini
	// bills thinking tokens as output tokens; CandidatesTokenCount alone would
	// under-report the billed output of thinking models.
	if resp.UsageMetadata != nil {
		response.InputToken = int(resp.UsageMetadata.PromptTokenCount)
		response.OutputToken = int(resp.UsageMetadata.CandidatesTokenCount) + int(resp.UsageMetadata.ThoughtsTokenCount)
		response.CacheReadInputToken = int(resp.UsageMetadata.CachedContentTokenCount)
	}

	for _, candidate := range resp.Candidates {
		if candidate.FinishReason != "" {
			// gollem does not set candidateCount, so the API returns one
			// candidate and the first reported reason is the reason for the
			// whole response.
			if response.FinishReason == "" {
				response.FinishReason = string(candidate.FinishReason)
			}
			if response.Refusal == nil {
				response.Refusal = candidateRefusal(candidate)
			}
		}

		if candidate.Content == nil {
			continue
		}

		for _, part := range candidate.Content.Parts {
			// Extract thought parts (internal reasoning from thinking models)
			if part.Thought {
				if part.Text != "" {
					response.Thoughts = append(response.Thoughts, part.Text)
				}
				continue
			}

			if part.Text != "" {
				response.Texts = append(response.Texts, part.Text)
			}

			if part.FunctionCall != nil {
				// Gemini 3.x returns FunctionCall.ID. Preserve it so the caller
				// can echo it back via FunctionResponse for strict matching.
				// For older models that leave it empty, synthesize a fallback
				// that the conversion layer will strip on the way back to
				// Gemini.
				id := part.FunctionCall.ID
				if id == "" {
					id = geminiFallbackToolCallID(part.FunctionCall.Name, len(response.FunctionCalls))
				}
				fc := &gollem.FunctionCall{
					ID:        id,
					Name:      part.FunctionCall.Name,
					Arguments: part.FunctionCall.Args,
				}
				response.FunctionCalls = append(response.FunctionCalls, fc)
			}
		}
	}

	return response
}

// candidateRefusal returns the refusal details of a candidate whose output was
// blocked, or nil when the finish reason does not report a block.
func candidateRefusal(candidate *genai.Candidate) *gollem.Refusal {
	switch candidate.FinishReason {
	case genai.FinishReasonSafety,
		genai.FinishReasonRecitation,
		genai.FinishReasonBlocklist,
		genai.FinishReasonProhibitedContent,
		genai.FinishReasonSPII,
		genai.FinishReasonImageSafety,
		genai.FinishReasonImageProhibitedContent,
		genai.FinishReasonImageRecitation:
		return &gollem.Refusal{
			Reason:      string(candidate.FinishReason),
			Categories:  blockedCategories(candidate.SafetyRatings),
			Explanation: candidate.FinishMessage,
		}
	default:
		return nil
	}
}

// promptRefusal returns the refusal details of a blocked prompt, or nil when
// the prompt was not blocked.
func promptRefusal(feedback *genai.GenerateContentResponsePromptFeedback) *gollem.Refusal {
	if feedback == nil || feedback.BlockReason == "" {
		return nil
	}
	return &gollem.Refusal{
		Reason:      string(feedback.BlockReason),
		Categories:  blockedCategories(feedback.SafetyRatings),
		Explanation: feedback.BlockReasonMessage,
	}
}

// blockedCategories returns the harm category of each safety rating that
// caused the block.
func blockedCategories(ratings []*genai.SafetyRating) []string {
	var categories []string
	for _, rating := range ratings {
		if rating != nil && rating.Blocked {
			categories = append(categories, string(rating.Category))
		}
	}
	return categories
}

// Generate generates content based on the input with optional per-call overrides.
func (s *Session) Generate(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
	// Build the content request for middleware
	// Create a copy of the current history to avoid middleware side effects
	// Always create history (even if empty) to maintain consistency with middleware
	historyCopy, err := newHistory(s.historyContents, s.issuer)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to convert history from Gemini format")
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
			s.historyContents, err = toContents(req.History, s.issuer)
			if err != nil {
				return nil, goerr.Wrap(err, "failed to convert history to Gemini format")
			}
		}

		// Build complete content list from history and inputs
		var contents []*genai.Content

		// Add history to contents if available
		if len(s.historyContents) > 0 {
			contents = append(contents, s.historyContents...)
		}

		// Convert current inputs to parts
		parts, err := s.convertInputs(req.Inputs...)
		if err != nil {
			return nil, err
		}

		// Add current input as a new user message.
		// newTurnContents tracks only the contents added in this turn so that
		// trace data records the delta rather than the full history.
		var newTurnContents []*genai.Content
		if len(parts) > 0 {
			userContent := &genai.Content{
				Role:  "user",
				Parts: parts,
			}
			contents = append(contents, userContent)
			newTurnContents = append(newTurnContents, userContent)
		}

		// Start LLM call trace span
		var geminiTraceData *trace.LLMCallData
		var llmErr error
		if h := trace.HandlerFrom(ctx); h != nil {
			ctx = h.StartLLMCall(ctx)
			defer func() { h.EndLLMCall(ctx, geminiTraceData, llmErr) }()
		}

		// Build effective config with per-call overrides
		effectiveConfig, err := s.buildEffectiveConfig(opts...)
		if err != nil {
			return nil, err
		}

		// Call the API
		result, err := s.apiClient.GenerateContent(ctx, s.model, contents, effectiveConfig)
		if err != nil {
			llmErr = err
			opts := tokenLimitErrorOptions(err)
			return nil, goerr.Wrap(err, "failed to generate content", opts...)
		}

		response, err := processResponse(result)
		if err != nil {
			llmErr = err
			// The span records only the error message; record the response as
			// well so that the finish reason and the refusal details are kept.
			geminiTraceData = buildGeminiTraceData(convertResponse(result), s.model, s.cfg.SystemPrompt(), newTurnContents)
			return nil, err
		}

		// Set trace data for defer.
		// Record only contents added in this turn; previous turns are already
		// captured in earlier trace spans.
		geminiTraceData = buildGeminiTraceData(response, s.model, s.cfg.SystemPrompt(), newTurnContents)

		// Update history with the input and the response content. The parts are
		// taken from the candidate itself so that ThoughtSignature survives.
		var newContents []*genai.Content
		// Add current input as a new user message
		if len(parts) > 0 {
			userContent := &genai.Content{
				Role:  "user",
				Parts: parts,
			}
			newContents = append(newContents, userContent)
		}
		// Add the assistant response content from candidates, normalized for
		// storage (thought text dropped, signatures kept).
		for _, candidate := range result.Candidates {
			if stored := newHistoryContent(candidate.Content); stored != nil {
				newContents = append(newContents, stored)
			}
		}

		// Append new contents to history
		if len(newContents) > 0 {
			s.historyContents = append(s.historyContents, newContents...)
		}

		return &gollem.ContentResponse{
			Texts:               response.Texts,
			Thoughts:            response.Thoughts,
			FunctionCalls:       response.FunctionCalls,
			InputToken:          response.InputToken,
			OutputToken:         response.OutputToken,
			CacheReadInputToken: response.CacheReadInputToken,
			FinishReason:        response.FinishReason,
			Refusal:             response.Refusal,
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
		Texts:               contentResp.Texts,
		Thoughts:            contentResp.Thoughts,
		FunctionCalls:       contentResp.FunctionCalls,
		InputToken:          contentResp.InputToken,
		OutputToken:         contentResp.OutputToken,
		CacheReadInputToken: contentResp.CacheReadInputToken,
		FinishReason:        contentResp.FinishReason,
		Refusal:             contentResp.Refusal,
	}, nil
}

// Stream generates content based on the input and returns a stream of responses with optional per-call overrides.
func (s *Session) Stream(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (<-chan *gollem.Response, error) {
	// Build the content request for middleware
	// Create a copy of the current history to avoid middleware side effects
	// Always create history (even if empty) to maintain consistency with middleware
	historyCopy, err := newHistory(s.historyContents, s.issuer)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to convert history from Gemini format")
	}

	contentReq := &gollem.ContentRequest{
		Inputs:       input,
		History:      historyCopy,
		SystemPrompt: s.cfg.SystemPrompt(),
	}

	// Create the base handler that performs the actual API call
	baseHandler := func(ctx context.Context, req *gollem.ContentRequest) (<-chan *gollem.ContentResponse, error) {
		// Always update history from middleware (even if same address, content may have changed)
		if req.History != nil {
			var err error
			s.historyContents, err = toContents(req.History, s.issuer)
			if err != nil {
				return nil, goerr.Wrap(err, "failed to convert history to Gemini format")
			}
		}

		// Build complete content list from history and inputs
		var contents []*genai.Content

		// Add history to contents if available
		if len(s.historyContents) > 0 {
			contents = append(contents, s.historyContents...)
		}

		// Convert current inputs to parts
		parts, err := s.convertInputs(req.Inputs...)
		if err != nil {
			return nil, err
		}

		// Add current input as a new user message.
		// newTurnContents tracks only the contents added in this turn so that
		// trace data records the delta rather than the full history.
		var newTurnContents []*genai.Content
		if len(parts) > 0 {
			userContent := &genai.Content{
				Role:  "user",
				Parts: parts,
			}
			contents = append(contents, userContent)
			newTurnContents = append(newTurnContents, userContent)
		}

		// Start LLM call trace span
		traceHandler := trace.HandlerFrom(ctx)
		if traceHandler != nil {
			ctx = traceHandler.StartLLMCall(ctx)
		}

		// Create streaming channel for middleware
		streamChan := make(chan *gollem.ContentResponse)

		// Start streaming in goroutine
		go func() {
			defer close(streamChan)

			var streamTraceData *trace.LLMCallData
			var streamErr error
			if traceHandler != nil {
				defer func() { traceHandler.EndLLMCall(ctx, streamTraceData, streamErr) }()
			}

			// Build effective config with per-call overrides
			effectiveConfig, err := s.buildEffectiveConfig(opts...)
			if err != nil {
				streamChan <- &gollem.ContentResponse{Error: err}
				return
			}

			// Get the streaming response from API
			apiStreamChan := s.apiClient.GenerateContentStream(ctx, s.model, contents, effectiveConfig)

			// Accumulate response data for history
			var accumulatedTexts, accumulatedThoughts []string
			var accumulatedFunctionCalls []*gollem.FunctionCall
			// accumulatedParts keeps the raw parts as returned by the API so
			// that ThoughtSignature and FunctionCall.ID survive into history.
			var accumulatedParts []*genai.Part
			var totalInputTokens, totalOutputTokens int
			var totalCacheRead int
			var finishReason string
			var refusal *gollem.Refusal

			for streamResp := range apiStreamChan {
				if streamResp.Err != nil {
					streamErr = streamResp.Err
					opts := tokenLimitErrorOptions(streamResp.Err)
					streamChan <- &gollem.ContentResponse{
						Error: goerr.Wrap(streamResp.Err, "failed to generate content stream", opts...),
					}
					return
				}

				// Process the response
				response, err := processResponse(streamResp.Resp)
				if err != nil {
					streamErr = err
					// The span records only the error message; record the
					// chunk as well so that the finish reason and the refusal
					// details are kept.
					streamTraceData = buildGeminiTraceData(convertResponse(streamResp.Resp), s.model, s.cfg.SystemPrompt(), newTurnContents)
					streamChan <- &gollem.ContentResponse{
						Error: err,
					}
					return
				}

				// Accumulate data. Usage counts are per-call snapshots (Gemini
				// reports the running total, not per-chunk deltas), so take the
				// latest non-zero value instead of summing.
				accumulatedTexts = append(accumulatedTexts, response.Texts...)
				accumulatedThoughts = append(accumulatedThoughts, response.Thoughts...)
				accumulatedFunctionCalls = append(accumulatedFunctionCalls, response.FunctionCalls...)
				for _, candidate := range streamResp.Resp.Candidates {
					if candidate.Content == nil {
						continue
					}
					accumulatedParts = append(accumulatedParts, candidate.Content.Parts...)
				}
				if response.InputToken > 0 {
					totalInputTokens = response.InputToken
				}
				if response.OutputToken > 0 {
					totalOutputTokens = response.OutputToken
				}
				if response.CacheReadInputToken > 0 {
					totalCacheRead = response.CacheReadInputToken
				}
				if response.FinishReason != "" {
					finishReason = response.FinishReason
				}
				if response.Refusal != nil {
					refusal = response.Refusal
				}

				// Send streaming response with the running totals
				streamChan <- &gollem.ContentResponse{
					Texts:               response.Texts,
					Thoughts:            response.Thoughts,
					FunctionCalls:       response.FunctionCalls,
					InputToken:          totalInputTokens,
					OutputToken:         totalOutputTokens,
					CacheReadInputToken: totalCacheRead,
					FinishReason:        response.FinishReason,
					Refusal:             response.Refusal,
				}
			}

			// Update history from the raw streamed parts. Rebuilding the parts
			// from accumulated texts and function calls would drop
			// ThoughtSignature and FunctionCall.ID, and Gemini 3.x rejects a
			// later turn whose function call parts lost their signature.
			//
			// The user turn is stored only together with a model turn: a
			// history ending on two consecutive user contents is rejected by
			// models that require alternating roles.
			// Normalize before merging so that text separated only by a dropped
			// part still collapses into a single history part.
			assistantContent := newHistoryContent(&genai.Content{
				Role:  "model",
				Parts: accumulatedParts,
			})
			if assistantContent != nil {
				assistantContent.Parts = mergeStreamedParts(assistantContent.Parts)

				var newContents []*genai.Content

				// Convert inputs to Gemini content
				inputParts, err := s.convertInputs(req.Inputs...)
				if err == nil && len(inputParts) > 0 {
					userContent := &genai.Content{
						Role:  "user",
						Parts: inputParts,
					}
					newContents = append(newContents, userContent)
				}
				newContents = append(newContents, assistantContent)

				s.historyContents = append(s.historyContents, newContents...)
			}

			// Set trace data for defer.
			// Record only contents added in this turn; previous turns are
			// already captured in earlier trace spans.
			streamTraceData = &trace.LLMCallData{
				InputTokens:          totalInputTokens,
				OutputTokens:         totalOutputTokens,
				Model:                s.model,
				CacheReadInputTokens: totalCacheRead,
				Request: &trace.LLMRequest{
					SystemPrompt: s.cfg.SystemPrompt(),
					Messages:     contentsToTraceMessages(newTurnContents),
				},
				Response: &trace.LLMResponse{
					FinishReason: finishReason,
					Refusal:      convert.TraceRefusal(refusal),
				},
			}
			if len(accumulatedTexts) > 0 {
				streamTraceData.Response.Texts = accumulatedTexts
			}
			if len(accumulatedThoughts) > 0 {
				streamTraceData.Response.Thoughts = accumulatedThoughts
			}
			for _, fc := range accumulatedFunctionCalls {
				streamTraceData.Response.FunctionCalls = append(streamTraceData.Response.FunctionCalls, &trace.FunctionCall{
					ID:        fc.ID,
					Name:      fc.Name,
					Arguments: fc.Arguments,
				})
			}
		}()

		return streamChan, nil
	}

	// Build middleware chain for streaming
	handler := gollem.ContentStreamHandler(baseHandler)
	for i := len(s.cfg.ContentStreamMiddlewares()) - 1; i >= 0; i-- {
		handler = s.cfg.ContentStreamMiddlewares()[i](handler)
	}

	// Execute middleware chain
	streamChan, err := handler(ctx, contentReq)
	if err != nil {
		return nil, err
	}

	// Convert ContentResponse stream to Response stream
	respChan := make(chan *gollem.Response)
	go func() {
		defer close(respChan)

		for contentResp := range streamChan {
			if contentResp.Error != nil {
				// Forward the error: dropping it here would end the stream with
				// no output and no error, which the caller cannot distinguish
				// from a model that simply said nothing.
				respChan <- &gollem.Response{Error: contentResp.Error}
				continue
			}

			// Convert ContentResponse to Response
			resp := &gollem.Response{
				Texts:               contentResp.Texts,
				Thoughts:            contentResp.Thoughts,
				FunctionCalls:       contentResp.FunctionCalls,
				InputToken:          contentResp.InputToken,
				OutputToken:         contentResp.OutputToken,
				CacheReadInputToken: contentResp.CacheReadInputToken,
				FinishReason:        contentResp.FinishReason,
				Refusal:             contentResp.Refusal,
			}

			respChan <- resp
		}
	}()

	return respChan, nil
}

// GenerateEmbedding generates embeddings for the given input texts.
func (c *Client) GenerateEmbedding(ctx context.Context, dimension int, input []string) ([][]float64, error) {
	// Create content for embedding
	contents := make([]*genai.Content, len(input))
	for i, text := range input {
		contents[i] = &genai.Content{
			Parts: []*genai.Part{
				{Text: text},
			},
		}
	}

	// Create embedding config
	config := &genai.EmbedContentConfig{}
	if dimension > 0 && dimension <= math.MaxInt32 {
		outputDim := int32(dimension)
		config.OutputDimensionality = &outputDim
	}

	// Start LLM call trace span
	var traceData *trace.LLMCallData
	var llmErr error
	if h := trace.HandlerFrom(ctx); h != nil {
		ctx = h.StartLLMCall(ctx)
		defer func() { h.EndLLMCall(ctx, traceData, llmErr) }()
	}

	// Generate embeddings for the specified model
	result, err := c.client.Models.EmbedContent(ctx, c.embeddingModel, contents, config)
	if err != nil {
		llmErr = err
		return nil, goerr.Wrap(err, "failed to generate embeddings")
	}

	traceData = &trace.LLMCallData{
		Model:    c.embeddingModel,
		Request:  &trace.LLMRequest{},
		Response: &trace.LLMResponse{},
	}

	if result == nil || len(result.Embeddings) == 0 {
		return nil, goerr.New("no embeddings returned")
	}

	embeddings := make([][]float64, len(result.Embeddings))
	for i, emb := range result.Embeddings {
		embeddings[i] = make([]float64, len(emb.Values))
		for j, v := range emb.Values {
			embeddings[i][j] = float64(v)
		}
	}

	return embeddings, nil
}

// Helper function to convert new SDK history to gollem.History

// convertToolToNewSDK converts gollem.Tool to new SDK's FunctionDeclaration
//
// genai.Schema (genai v1.53.0) has no field for additionalProperties, so a tool
// whose parameters contain a map (gollem.Parameter.AdditionalProperties) is
// sent as JSON Schema in ParametersJsonSchema, which accepts it. Other tools
// keep the genai.Schema form so that their requests do not change.
func convertToolToNewSDK(tool gollem.Tool) *genai.FunctionDeclaration {
	spec := tool.Spec()

	root := &gollem.Parameter{Type: gollem.TypeObject, Properties: spec.Parameters}
	if _, found := gollemschema.FindAdditionalProperties(root); found {
		return &genai.FunctionDeclaration{
			Name:                 spec.Name,
			Description:          spec.Description,
			ParametersJsonSchema: gollemschema.ConvertParameterToJSONSchema(root),
		}
	}

	// Collect required fields from parameters
	required := gollemschema.CollectRequiredFields(spec.Parameters)
	if required == nil {
		required = []string{}
	}

	parameters := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: make(map[string]*genai.Schema),
		Required:   required,
	}

	for name, param := range spec.Parameters {
		parameters.Properties[name] = convertParameterToNewSchema(param)
	}

	return &genai.FunctionDeclaration{
		Name:        spec.Name,
		Description: spec.Description,
		Parameters:  parameters,
	}
}

// convertParameterToNewSchema converts gollem.Parameter to new SDK's schema
//
// A union is sent in genai.Schema.AnyOf with Type left unset. The Vertex AI
// documentation for controlled generation lists anyOf among the supported
// responseSchema fields, and function declaration parameters use the same
// Schema type.
func convertParameterToNewSchema(param *gollem.Parameter) *genai.Schema {
	if len(param.AnyOf) > 0 {
		return &genai.Schema{
			AnyOf:       gollemschema.ConvertAnyOf(param.AnyOf, convertParameterToNewSchema),
			Description: param.Description,
			Title:       param.Title,
		}
	}

	schema := &genai.Schema{
		Type:        getNewGeminiType(param.Type),
		Description: param.Description,
		Title:       param.Title,
	}

	if len(param.Enum) > 0 {
		schema.Enum = param.Enum
	}

	if param.Properties != nil {
		schema.Properties = make(map[string]*genai.Schema)
		for name, prop := range param.Properties {
			schema.Properties[name] = convertParameterToNewSchema(prop)
		}
		if required := gollemschema.CollectRequiredFields(param.Properties); len(required) > 0 {
			schema.Required = required
		} else {
			schema.Required = []string{}
		}
	}

	if param.Items != nil {
		schema.Items = convertParameterToNewSchema(param.Items)
	}

	// Add number constraints
	if param.Type == gollem.TypeNumber || param.Type == gollem.TypeInteger {
		if param.Minimum != nil {
			minVal := *param.Minimum
			schema.Minimum = &minVal
		}
		if param.Maximum != nil {
			maxVal := *param.Maximum
			schema.Maximum = &maxVal
		}
	}

	// Add string constraints
	if param.Type == gollem.TypeString {
		if param.MinLength != nil {
			minLen := int64(*param.MinLength)
			schema.MinLength = &minLen
		}
		if param.MaxLength != nil {
			maxLen := int64(*param.MaxLength)
			schema.MaxLength = &maxLen
		}
		if param.Pattern != "" {
			schema.Pattern = param.Pattern
		}
	}

	// Add array constraints
	if param.Type == gollem.TypeArray {
		if param.MinItems != nil {
			minItems := int64(*param.MinItems)
			schema.MinItems = &minItems
		}
		if param.MaxItems != nil {
			maxItems := int64(*param.MaxItems)
			schema.MaxItems = &maxItems
		}
	}

	return schema
}

func getNewGeminiType(paramType gollem.ParameterType) genai.Type {
	switch paramType {
	case gollem.TypeString:
		return genai.TypeString
	case gollem.TypeNumber:
		return genai.TypeNumber
	case gollem.TypeInteger:
		return genai.TypeInteger
	case gollem.TypeBoolean:
		return genai.TypeBoolean
	case gollem.TypeArray:
		return genai.TypeArray
	case gollem.TypeObject:
		return genai.TypeObject
	default:
		return genai.TypeString
	}
}

// setResponseSchema validates param and stores it in config.
//
// genai.Schema (genai v1.53.0) has no field for additionalProperties, so a
// schema that contains a map (gollem.Parameter.AdditionalProperties) is sent as
// JSON Schema in ResponseJsonSchema, which accepts it. Any other schema keeps
// the genai.Schema form in ResponseSchema so that its requests do not change.
// The unused field is cleared so that a per-call schema never leaves the
// session's schema of the other form in the request.
func setResponseSchema(config *genai.GenerateContentConfig, param *gollem.Parameter) error {
	if _, found := gollemschema.FindAdditionalProperties(param); found {
		if err := param.Validate(); err != nil {
			return goerr.Wrap(err, "invalid response schema")
		}
		config.ResponseJsonSchema = gollemschema.ConvertParameterToJSONSchema(param)
		config.ResponseSchema = nil
		return nil
	}

	schema, err := convertResponseSchemaToGenai(param)
	if err != nil {
		return err
	}
	config.ResponseSchema = schema
	config.ResponseJsonSchema = nil
	return nil
}

// convertResponseSchemaToGenai converts gollem.Parameter to genai.Schema
func convertResponseSchemaToGenai(param *gollem.Parameter) (*genai.Schema, error) {
	if param == nil {
		return nil, nil
	}

	// Validate schema first
	if err := param.Validate(); err != nil {
		return nil, goerr.Wrap(err, "invalid response schema")
	}

	// Convert using existing convertParameterToNewSchema function
	schema := convertParameterToNewSchema(param)

	return schema, nil
}

// buildEffectiveConfig creates a copy of the session config with per-call overrides applied.
func (s *Session) buildEffectiveConfig(opts ...gollem.GenerateOption) (*genai.GenerateContentConfig, error) {
	genCfg := gollem.NewGenerateConfig(opts...)
	effectiveConfig := *s.config
	if m := genCfg.MaxTokens(); m != nil {
		if *m > math.MaxInt32 || *m < 0 {
			return nil, goerr.New("maxTokens out of int32 range", goerr.V("maxTokens", *m))
		}
		effectiveConfig.MaxOutputTokens = int32(*m)
	}
	if perCallSchema := genCfg.ResponseSchema(); perCallSchema != nil {
		effectiveConfig.ResponseMIMEType = "application/json"
		if err := setResponseSchema(&effectiveConfig, perCallSchema); err != nil {
			return nil, goerr.Wrap(err, "failed to convert per-call response schema")
		}
	}
	// The tool declarations stay in the request; mode NONE only forbids calling
	// them. Without tools there is nothing to forbid, so no ToolConfig is sent.
	if genCfg.ToolCallsDisabled() && len(effectiveConfig.Tools) > 0 {
		effectiveConfig.ToolConfig = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeNone,
			},
		}
	}
	return &effectiveConfig, nil
}

// CountToken calculates the total number of tokens for the given inputs,
// including system prompt, history messages, and new inputs.
// This is useful for estimating API costs and checking token limits before making actual API calls.
func (s *Session) CountToken(ctx context.Context, input ...gollem.Input) (int, error) {
	// Build complete content list from history and inputs
	var contents []*genai.Content

	// Create a copy of history contents to avoid race conditions
	// This ensures thread safety when reading historyContents
	if len(s.historyContents) > 0 {
		historyContentsCopy := make([]*genai.Content, len(s.historyContents))
		copy(historyContentsCopy, s.historyContents)
		contents = append(contents, historyContentsCopy...)
	}

	// Convert current inputs to parts
	parts, err := s.convertInputs(input...)
	if err != nil {
		return 0, goerr.Wrap(err, "failed to convert inputs for token counting")
	}

	// Add current input as a new user message
	if len(parts) > 0 {
		userContent := &genai.Content{
			Role:  "user",
			Parts: parts,
		}
		contents = append(contents, userContent)
	}

	// Create CountTokensConfig from GenerateContentConfig
	// Note: config fields are read-only after session creation, so no deep copy needed
	countConfig := &genai.CountTokensConfig{
		SystemInstruction: s.config.SystemInstruction,
		Tools:             s.config.Tools,
	}

	// Start LLM call trace span
	var traceData *trace.LLMCallData
	var llmErr error
	if h := trace.HandlerFrom(ctx); h != nil {
		ctx = h.StartLLMCall(ctx)
		defer func() { h.EndLLMCall(ctx, traceData, llmErr) }()
	}

	// Call the CountTokens API
	result, err := s.apiClient.CountTokens(ctx, s.model, contents, countConfig)
	if err != nil {
		llmErr = err
		return 0, goerr.Wrap(err, "failed to count tokens")
	}

	traceData = &trace.LLMCallData{
		InputTokens: int(result.TotalTokens),
		Model:       s.model,
		Request:     &trace.LLMRequest{},
		Response:    &trace.LLMResponse{},
	}

	return int(result.TotalTokens), nil
}

// tokenLimitErrorOptions checks if the error is a token limit exceeded error
// and returns goerr.Option to tag the error with ErrTagTokenExceeded.
// Returns nil if the error is not a token limit exceeded error.
//
// Detection logic:
// - Error must be *genai.APIError
// - Code must be 400 and Status must be "INVALID_ARGUMENT"
// - Message must have one of these prefixes:
//   - "The model's maximum context length is"
//   - "Request payload size exceeds the limit"
func tokenLimitErrorOptions(err error) []goerr.Option {
	var apiErr *genai.APIError
	if !errors.As(err, &apiErr) {
		return nil
	}

	if apiErr.Code != 400 || apiErr.Status != "INVALID_ARGUMENT" {
		return nil
	}

	if strings.HasPrefix(apiErr.Message, "The model's maximum context length is") ||
		strings.HasPrefix(apiErr.Message, "Request payload size exceeds the limit") {
		return []goerr.Option{goerr.Tag(gollem.ErrTagTokenExceeded)}
	}

	return nil
}

// contentsToTraceMessages converts Gemini contents to trace messages.
func contentsToTraceMessages(contents []*genai.Content) []trace.Message {
	var messages []trace.Message
	for _, c := range contents {
		if c == nil {
			continue
		}
		var blocks []trace.MessageContent
		for _, p := range c.Parts {
			if p == nil {
				continue
			}
			switch {
			case p.Thought && p.Text != "":
				blocks = append(blocks, trace.NewThinkingContent(p.Text))
			case p.Text != "":
				blocks = append(blocks, trace.NewTextContent(p.Text))
			case p.FunctionCall != nil:
				blocks = append(blocks, trace.NewToolCallContent(
					"", p.FunctionCall.Name, p.FunctionCall.Args,
				))
			case p.FunctionResponse != nil:
				var result map[string]any
				if p.FunctionResponse.Response != nil {
					result = p.FunctionResponse.Response
				}
				blocks = append(blocks, trace.NewToolResponseContent(
					"", p.FunctionResponse.Name, result,
				))
			case p.InlineData != nil:
				mc := trace.NewMediaContent("image", p.InlineData.MIMEType)
				if p.InlineData.DisplayName != "" {
					mc.Title = p.InlineData.DisplayName
				}
				blocks = append(blocks, mc)
			case p.FileData != nil:
				mc := trace.NewMediaContent("file", p.FileData.MIMEType)
				mc.URL = p.FileData.FileURI
				if p.FileData.DisplayName != "" {
					mc.Title = p.FileData.DisplayName
				}
				blocks = append(blocks, mc)
			case p.ExecutableCode != nil:
				blocks = append(blocks, trace.MessageContent{
					Type: "executable_code",
					Text: p.ExecutableCode.Code,
				})
			case p.CodeExecutionResult != nil:
				blocks = append(blocks, trace.MessageContent{
					Type: "code_execution_result",
					Text: p.CodeExecutionResult.Output,
				})
			}
		}
		if len(blocks) > 0 {
			messages = append(messages, trace.Message{
				Role:     c.Role,
				Contents: blocks,
			})
		}
	}
	return messages
}

// buildGeminiTraceData builds trace.LLMCallData from a processed gollem.Response.
func buildGeminiTraceData(response *gollem.Response, model string, systemPrompt string, contents []*genai.Content) *trace.LLMCallData {
	data := &trace.LLMCallData{
		InputTokens:          response.InputToken,
		OutputTokens:         response.OutputToken,
		Model:                model,
		CacheReadInputTokens: response.CacheReadInputToken,
		Request: &trace.LLMRequest{
			SystemPrompt: systemPrompt,
			Messages:     contentsToTraceMessages(contents),
		},
		Response: &trace.LLMResponse{
			FinishReason: response.FinishReason,
			Refusal:      convert.TraceRefusal(response.Refusal),
		},
	}

	if len(response.Texts) > 0 {
		data.Response.Texts = response.Texts
	}
	if len(response.Thoughts) > 0 {
		data.Response.Thoughts = response.Thoughts
	}
	for _, fc := range response.FunctionCalls {
		data.Response.FunctionCalls = append(data.Response.FunctionCalls, &trace.FunctionCall{
			ID:        fc.ID,
			Name:      fc.Name,
			Arguments: fc.Arguments,
		})
	}

	return data
}
