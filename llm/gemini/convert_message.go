package gemini

import (
	"encoding/json"

	"github.com/m-mizutani/goerr/v2"
	"github.com/m-mizutani/gollem"
	"github.com/m-mizutani/gollem/internal/convert"
	"google.golang.org/genai"
)

// partMeta is the metadata stored in MessageContent.Meta for Gemini parts.
// It preserves Gemini-specific fields (e.g., thinking model signatures) across
// serialization/deserialization without polluting the common message types.
type partMeta struct {
	Thought          bool   `json:"thought,omitempty"`
	ThoughtSignature []byte `json:"thought_signature,omitempty"`
}

// marshalPartMeta marshals partMeta to JSON for MessageContent.Meta.
// Returns nil if no metadata needs to be stored.
func marshalPartMeta(m partMeta) (json.RawMessage, error) {
	if !m.Thought && len(m.ThoughtSignature) == 0 {
		return nil, nil
	}
	data, err := json.Marshal(m)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to marshal part meta")
	}
	return data, nil
}

// unmarshalPartMeta unmarshals partMeta from MessageContent.Meta.
// Returns zero-value partMeta if meta is nil or empty.
func unmarshalPartMeta(meta json.RawMessage) (partMeta, error) {
	if len(meta) == 0 {
		return partMeta{}, nil
	}
	var m partMeta
	if err := json.Unmarshal(meta, &m); err != nil {
		return partMeta{}, goerr.Wrap(err, "failed to unmarshal part meta")
	}
	return m, nil
}

// isEmptyPart returns true if a Gemini part has no meaningful content.
// Some models (e.g., thinking models) may return empty parts that should be skipped.
func isEmptyPart(part *genai.Part) bool {
	return part.Text == "" &&
		!part.Thought &&
		part.FunctionCall == nil &&
		part.FunctionResponse == nil &&
		part.InlineData == nil &&
		part.FileData == nil &&
		part.ExecutableCode == nil &&
		part.CodeExecutionResult == nil &&
		len(part.ThoughtSignature) == 0
}

// filterEmptyParts removes empty parts from a Content and returns a new Content.
// Returns nil if all parts are empty.
func filterEmptyParts(content *genai.Content) *genai.Content {
	if content == nil {
		return nil
	}
	filtered := make([]*genai.Part, 0, len(content.Parts))
	for _, part := range content.Parts {
		if !isEmptyPart(part) {
			filtered = append(filtered, part)
		}
	}
	if len(filtered) == 0 {
		return nil
	}
	return &genai.Content{
		Role:  content.Role,
		Parts: filtered,
	}
}

// convertGeminiToMessages converts Gemini contents to common Message format
func convertGeminiToMessages(contents []*genai.Content) ([]gollem.Message, error) {
	if len(contents) == 0 {
		return []gollem.Message{}, nil
	}

	result := make([]gollem.Message, 0, len(contents))

	for _, content := range contents {
		msg, err := convertGeminiContent(content)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert Gemini content")
		}
		result = append(result, msg)
	}

	return result, nil
}

// convertGeminiContent converts a single Gemini content to Message
func convertGeminiContent(content *genai.Content) (gollem.Message, error) {
	contents := make([]gollem.MessageContent, 0, len(content.Parts))

	for _, part := range content.Parts {
		if isEmptyPart(part) {
			continue
		}
		msgContent, err := convertGeminiPart(part)
		if err != nil {
			return gollem.Message{}, goerr.Wrap(err, "failed to convert Gemini part")
		}
		contents = append(contents, msgContent)
	}

	// Convert role - model is always converted to assistant
	role := convert.ConvertRoleToCommon(content.Role)

	return gollem.Message{
		Role:     role,
		Contents: contents,
	}, nil
}

// convertGeminiPart converts a Gemini part to MessageContent
func convertGeminiPart(part *genai.Part) (gollem.MessageContent, error) {
	// Build metadata from thinking-related fields
	meta, err := marshalPartMeta(partMeta{
		Thought:          part.Thought,
		ThoughtSignature: part.ThoughtSignature,
	})
	if err != nil {
		return gollem.MessageContent{}, err
	}

	// Thought part (internal reasoning, may have empty text)
	if part.Thought {
		mc, err := gollem.NewThinkingContent(part.Text)
		if err != nil {
			return gollem.MessageContent{}, err
		}
		mc.Meta = meta
		return mc, nil
	}

	// Text part
	if part.Text != "" {
		mc, err := gollem.NewTextContent(part.Text)
		if err != nil {
			return gollem.MessageContent{}, err
		}
		mc.Meta = meta
		return mc, nil
	}

	// Inline data (image or PDF)
	if part.InlineData != nil {
		if part.InlineData.MIMEType == "application/pdf" {
			return gollem.NewPDFContent(part.InlineData.Data, "")
		}
		return gollem.NewImageContent(
			part.InlineData.MIMEType,
			part.InlineData.Data,
			"",
			"",
		)
	}

	// File data
	if part.FileData != nil {
		// Gemini uses file URIs, store as URL
		return gollem.NewImageContent(
			part.FileData.MIMEType,
			nil,
			part.FileData.FileURI,
			"",
		)
	}

	// Function call
	if part.FunctionCall != nil {
		// Gemini 3.x returns FunctionCall.ID and requires strict id matching on the
		// corresponding FunctionResponse. Older Gemini models leave ID empty, so fall
		// back to a deterministic id derived from the function name.
		id := part.FunctionCall.ID
		if id == "" {
			id = convert.GenerateToolCallID(part.FunctionCall.Name, 0)
		}
		mc, err := gollem.NewToolCallContent(
			id,
			part.FunctionCall.Name,
			part.FunctionCall.Args,
		)
		if err != nil {
			return gollem.MessageContent{}, err
		}
		mc.Meta = meta
		return mc, nil
	}

	// Function response
	if part.FunctionResponse != nil {
		id := part.FunctionResponse.ID
		if id == "" {
			id = convert.GenerateToolCallID(part.FunctionResponse.Name, 0)
		}
		return gollem.NewToolResponseContent(
			id,
			part.FunctionResponse.Name,
			part.FunctionResponse.Response,
			false,
		)
	}

	// ThoughtSignature-only part (no text, no function call, not marked as thought)
	// Some Gemini models return parts with only ThoughtSignature set.
	if len(part.ThoughtSignature) > 0 {
		mc, err := gollem.NewTextContent("")
		if err != nil {
			return gollem.MessageContent{}, err
		}
		mc.Meta = meta
		return mc, nil
	}

	return gollem.MessageContent{}, goerr.Wrap(convert.ErrUnsupportedContentType, "unknown Gemini part type")
}

// convertMessagesToGemini converts common Messages to Gemini format
func convertMessagesToGemini(messages []gollem.Message) ([]*genai.Content, error) {
	if len(messages) == 0 {
		return []*genai.Content{}, nil
	}

	// Handle system messages by merging into first user message
	messages = convert.MergeSystemIntoFirstUser(messages)

	result := make([]*genai.Content, 0, len(messages))

	for _, msg := range messages {
		// Skip system messages as they've been merged
		if msg.Role == gollem.RoleSystem {
			continue
		}

		geminiContent, err := convertMessageToGemini(msg)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert message to Gemini format")
		}
		result = append(result, geminiContent)
	}

	return result, nil
}

// convertMessageToGemini converts a single Message to Gemini format
func convertMessageToGemini(msg gollem.Message) (*genai.Content, error) {
	// Convert role
	role := ""
	switch msg.Role {
	case gollem.RoleUser:
		role = "user"
	case gollem.RoleAssistant:
		// Assistant is always converted to model for Gemini SDK
		role = "model"
	case gollem.RoleTool:
		// Tool responses go to user role in Gemini
		role = "user"
	default:
		role = "user"
	}

	// Convert contents
	parts := make([]*genai.Part, 0, len(msg.Contents))
	for _, content := range msg.Contents {
		part, err := convertContentToGemini(content)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert content to Gemini format")
		}
		parts = append(parts, part)
	}

	return &genai.Content{
		Role:  role,
		Parts: parts,
	}, nil
}

// convertContentToGemini converts MessageContent to Gemini part
func convertContentToGemini(content gollem.MessageContent) (*genai.Part, error) {
	// Extract provider metadata if present
	meta, err := unmarshalPartMeta(content.Meta)
	if err != nil {
		return nil, err
	}

	switch content.Type {
	case gollem.MessageContentTypeText:
		textContent, err := content.GetTextContent()
		if err != nil {
			return nil, err
		}
		return &genai.Part{
			Text:             textContent.Text,
			Thought:          meta.Thought,
			ThoughtSignature: meta.ThoughtSignature,
		}, nil

	case gollem.MessageContentTypeThinking:
		reasoningContent, err := content.GetThinkingContent()
		if err != nil {
			return nil, err
		}
		return &genai.Part{
			Text:             reasoningContent.Text,
			Thought:          true,
			ThoughtSignature: meta.ThoughtSignature,
		}, nil

	case gollem.MessageContentTypeImage:
		imgContent, err := content.GetImageContent()
		if err != nil {
			return nil, err
		}
		if len(imgContent.Data) > 0 {
			// Inline data
			return &genai.Part{
				InlineData: &genai.Blob{
					MIMEType: imgContent.MediaType,
					Data:     imgContent.Data,
				},
			}, nil
		} else if imgContent.URL != "" {
			// File URI
			return &genai.Part{
				FileData: &genai.FileData{
					MIMEType: imgContent.MediaType,
					FileURI:  imgContent.URL,
				},
			}, nil
		}
		return nil, goerr.Wrap(convert.ErrInvalidMessageFormat, "image has neither data nor URL")

	case gollem.MessageContentTypePDF:
		pdfContent, err := content.GetPDFContent()
		if err != nil {
			return nil, err
		}
		if len(pdfContent.Data) > 0 {
			return &genai.Part{
				InlineData: &genai.Blob{
					MIMEType: "application/pdf",
					Data:     pdfContent.Data,
				},
			}, nil
		}
		if pdfContent.URL != "" {
			return &genai.Part{
				FileData: &genai.FileData{
					MIMEType: "application/pdf",
					FileURI:  pdfContent.URL,
				},
			}, nil
		}
		return nil, goerr.Wrap(convert.ErrInvalidMessageFormat, "PDF has neither data nor URL")

	case gollem.MessageContentTypeToolCall:
		toolCall, err := content.GetToolCallContent()
		if err != nil {
			return nil, err
		}
		// Skip the synthesized fallback id; it is internal to gollem and would
		// not match anything Gemini 3.x issued, causing strict-match failures.
		id := toolCall.ID
		if id == convert.GenerateToolCallID(toolCall.Name, 0) {
			id = ""
		}
		return &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   id,
				Name: toolCall.Name,
				Args: toolCall.Arguments,
			},
			ThoughtSignature: meta.ThoughtSignature,
		}, nil

	case gollem.MessageContentTypeToolResponse:
		toolResp, err := content.GetToolResponseContent()
		if err != nil {
			return nil, err
		}
		id := toolResp.ToolCallID
		if id == convert.GenerateToolCallID(toolResp.Name, 0) {
			id = ""
		}
		return &genai.Part{
			FunctionResponse: &genai.FunctionResponse{
				ID:       id,
				Name:     toolResp.Name,
				Response: toolResp.Response,
			},
		}, nil

	default:
		return nil, goerr.Wrap(convert.ErrUnsupportedContentType, "unsupported content type for Gemini", goerr.Value("type", content.Type))
	}
}

// ToContents converts gollem.History to Gemini contents
func ToContents(h *gollem.History) ([]*genai.Content, error) {
	if h == nil || len(h.Messages) == 0 {
		return []*genai.Content{}, nil
	}
	return convertMessagesToGemini(h.Messages)
}

// NewHistory creates gollem.History from Gemini contents
func NewHistory(contents []*genai.Content) (*gollem.History, error) {
	commonMessages, err := convertGeminiToMessages(contents)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to convert Gemini messages to common format")
	}

	return &gollem.History{
		LLType:   gollem.LLMTypeGemini,
		Version:  gollem.HistoryVersion,
		Messages: commonMessages,
	}, nil
}
