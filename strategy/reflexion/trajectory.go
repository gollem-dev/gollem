package reflexion

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/gollem-dev/gollem"
)

// buildTrajectory constructs a Trajectory from the current execution state.
func buildTrajectory(ctx context.Context, state *gollem.StrategyState, trialNum int) *Trajectory {
	// Get history from session
	history, err := state.Session.History()
	if err != nil {
		// If we can't get history, create a trajectory with error info
		history = &gollem.History{}
	}

	finalResponse := []string{}
	if state.LastResponse != nil {
		finalResponse = state.LastResponse.Texts
	}

	return &Trajectory{
		TrialNum:      trialNum,
		UserInputs:    state.InitInput,
		History:       history,
		FinalResponse: finalResponse,
		EndTime:       time.Now(),
	}
}

// formatUserInputs formats user inputs for display in prompts.
func formatUserInputs(inputs []gollem.Input) string {
	var parts []string
	for _, input := range inputs {
		parts = append(parts, input.String())
	}
	return strings.Join(parts, "\n")
}

// formatHistory formats conversation history for display in prompts.
func formatHistory(history *gollem.History) string {
	if history == nil {
		return "(no history)"
	}

	messages := history.Messages
	var parts []string

	for i, msg := range messages {
		role := msg.Role
		content := formatMessageContents(msg.Contents)
		parts = append(parts, fmt.Sprintf("[%d] %s: %s", i+1, role, content))
	}

	if len(parts) == 0 {
		return "(empty history)"
	}

	return strings.Join(parts, "\n")
}

// formatMessageContents formats message contents into a readable string.
func formatMessageContents(contents []gollem.MessageContent) string {
	var parts []string

	for _, content := range contents {
		switch content.Type {
		case gollem.MessageContentTypeText:
			// Unmarshal text content
			var tc gollem.TextContent
			if err := json.Unmarshal(content.Data, &tc); err != nil {
				parts = append(parts, fmt.Sprintf("[unmarshal error: %v]", err))
			} else {
				parts = append(parts, tc.Text)
			}
		case gollem.MessageContentTypeToolCall:
			// Unmarshal tool call
			var tc gollem.ToolCallContent
			if err := json.Unmarshal(content.Data, &tc); err != nil {
				parts = append(parts, fmt.Sprintf("[unmarshal error: %v]", err))
			} else {
				parts = append(parts, fmt.Sprintf("[Tool Call: %s]", tc.Name))
			}
		case gollem.MessageContentTypeToolResponse:
			// Unmarshal tool response
			var tr gollem.ToolResponseContent
			if err := json.Unmarshal(content.Data, &tr); err != nil {
				parts = append(parts, fmt.Sprintf("[unmarshal error: %v]", err))
			} else {
				parts = append(parts, fmt.Sprintf("[Tool Response: %s]", tr.ToolCallID))
			}
		}
	}

	if len(parts) == 0 {
		return "(empty)"
	}

	return strings.Join(parts, " ")
}
