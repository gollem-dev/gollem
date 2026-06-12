package convert

import (
	"encoding/json"
	"errors"
	"strconv"

	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/goerr/v2"
)

// Conversion errors
var (
	// ErrUnsupportedContentType is returned when a content type cannot be converted
	ErrUnsupportedContentType = errors.New("unsupported content type")

	// ErrInvalidMessageFormat is returned when a message has an invalid format
	ErrInvalidMessageFormat = errors.New("invalid message format")

	// ErrConversionFailed is returned when conversion between formats fails
	ErrConversionFailed = errors.New("conversion failed")
)

// ConversionWarning represents a warning during conversion
type ConversionWarning struct {
	Message string
	Field   string
	Value   interface{}
}

// ParseJSONArguments attempts to parse a JSON string into a map
func ParseJSONArguments(jsonStr string) (map[string]interface{}, error) {
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &args); err != nil {
		return nil, goerr.Wrap(err, "failed to parse JSON arguments")
	}
	return args, nil
}

// StringifyJSONArguments converts a map to a JSON string
func StringifyJSONArguments(args map[string]interface{}) (string, error) {
	data, err := json.Marshal(args)
	if err != nil {
		return "", goerr.Wrap(err, "failed to stringify JSON arguments")
	}
	return string(data), nil
}

// ConvertRoleToCommon converts various provider roles to common MessageRole
func ConvertRoleToCommon(role string) gollem.MessageRole {
	switch role {
	case "system":
		return gollem.RoleSystem
	case "user":
		return gollem.RoleUser
	case "assistant":
		return gollem.RoleAssistant
	case "tool":
		return gollem.RoleTool
	case "function":
		// Legacy OpenAI function role is treated as tool
		return gollem.RoleTool
	case "model":
		// Gemini's model role is treated as assistant
		return gollem.RoleAssistant
	default:
		// Default to user role for unknown roles
		return gollem.RoleUser
	}
}

// MergeSystemIntoFirstUser merges a system message into the first user message
// This is used for providers that don't support system messages directly (Claude, Gemini)
func MergeSystemIntoFirstUser(messages []gollem.Message) []gollem.Message {
	if len(messages) == 0 {
		return messages
	}

	// Find the first system message
	var systemContent string
	hasSystem := false
	for i, msg := range messages {
		if msg.Role == gollem.RoleSystem {
			hasSystem = true
			// Extract text content from system message
			for _, content := range msg.Contents {
				if content.Type == gollem.MessageContentTypeText {
					var textContent gollem.TextContent
					if err := json.Unmarshal(content.Data, &textContent); err == nil {
						if systemContent != "" {
							systemContent += "\n"
						}
						systemContent += textContent.Text
					}
				}
			}
			// Remove system message from the list
			messages = append(messages[:i], messages[i+1:]...)
			break
		}
	}

	if !hasSystem || systemContent == "" {
		return messages
	}

	// Find first user message and prepend system content
	for i, msg := range messages {
		if msg.Role == gollem.RoleUser {
			// Prepend system content to first user message
			newContent := make([]gollem.MessageContent, 0, len(msg.Contents)+1)

			// Add system content first
			if textContent, err := gollem.NewTextContent(systemContent + "\n\n"); err == nil {
				newContent = append(newContent, textContent)
			}

			// Add existing user content
			newContent = append(newContent, msg.Contents...)

			messages[i].Contents = newContent
			break
		}
	}

	return messages
}

// GenerateToolCallID generates a unique ID for tool calls if not present
func GenerateToolCallID(name string, index int) string {
	return "call_" + name + "_" + strconv.Itoa(index)
}
