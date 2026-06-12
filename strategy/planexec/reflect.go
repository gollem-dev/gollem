package planexec

import (
	"context"
	"encoding/json"

	"github.com/google/uuid"

	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/goerr/v2"
)

// reflectionResult holds the result of reflection
type reflectionResult struct {
	UpdatedTasks []Task // Modified tasks
	NewTasks     []Task // New tasks to add
}

// reflect performs reflection after task completion to update or add tasks
// It evaluates task results against the Plan, which contains all necessary context and constraints.
// This is an internal analysis process - the conversation history is not preserved
func reflect(ctx context.Context, client gollem.LLMClient, plan *Plan, completedTask *Task, tools []gollem.Tool, middleware []gollem.ContentBlockMiddleware, currentIteration, maxIterations int, history *gollem.History, systemPrompt string) (*reflectionResult, error) {
	// Create a new session for reflection with JSON content type
	// NOTE: Do NOT pass tools to reflection session.
	// - Tools: When provided, some LLM providers (like Gemini) prioritize function calls
	//   over JSON text responses, which breaks the reflection phase
	// - System prompt: Provided if available - helps with domain-specific reflection
	// - History: MUST be provided so reflection can detect already-executed tools
	sessionOpts := []gollem.SessionOption{
		gollem.WithSessionContentType(gollem.ContentTypeJSON),
	}
	if systemPrompt != "" {
		sessionOpts = append(sessionOpts, gollem.WithSessionSystemPrompt(systemPrompt))
	}
	if history != nil {
		sessionOpts = append(sessionOpts, gollem.WithSessionHistory(history))
	}
	for _, mw := range middleware {
		sessionOpts = append(sessionOpts, gollem.WithSessionContentBlockMiddleware(mw))
	}

	session, err := client.NewSession(ctx, sessionOpts...)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to create session for reflection")
	}

	// Build reflection prompt
	reflectPrompt := buildReflectPrompt(ctx, plan, completedTask.Result, tools, currentIteration, maxIterations)

	// Generate reflection using LLM
	response, err := session.Generate(ctx, reflectPrompt)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to generate reflection")
	}

	// Parse the reflection response
	result, err := parseReflectionFromResponse(ctx, response, plan)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to parse reflection response")
	}

	return result, nil
}

// parseReflectionFromResponse extracts reflection results from LLM response
func parseReflectionFromResponse(ctx context.Context, response *gollem.Response, currentPlan *Plan) (*reflectionResult, error) {
	result := &reflectionResult{
		UpdatedTasks: []Task{},
		NewTasks:     []Task{},
	}

	if response == nil {
		// If no response, return empty result (no updates)
		return result, nil
	}

	if len(response.Texts) == 0 {
		// If no text but has function calls, return empty result (no updates)
		return result, nil
	}

	// Parse JSON response directly (WithSessionContentType ensures JSON format)
	var reflectionResponse struct {
		NewTasks     []string `json:"new_tasks"` // Task descriptions for new tasks
		UpdatedTasks []struct {
			ID          string `json:"id"`
			Description string `json:"description"`
			State       string `json:"state"`
		} `json:"updated_tasks"` // Tasks to update (mark as failed, pending, etc.)
		Reason string `json:"reason"` // Explanation
	}

	if err := json.Unmarshal([]byte(response.Texts[0]), &reflectionResponse); err != nil {
		// If JSON parsing fails, return empty result
		return result, nil
	}

	// Process new tasks
	for _, taskDesc := range reflectionResponse.NewTasks {
		result.NewTasks = append(result.NewTasks, Task{
			ID:          uuid.New().String(),
			Description: taskDesc,
			State:       TaskStatePending,
		})
	}

	// Process updated tasks
	for _, updatedTask := range reflectionResponse.UpdatedTasks {
		state := TaskStatePending
		switch updatedTask.State {
		case "pending":
			state = TaskStatePending
		case "in_progress":
			state = TaskStateInProgress
		case "completed":
			state = TaskStateCompleted
		case "skipped":
			state = TaskStateSkipped
		}

		result.UpdatedTasks = append(result.UpdatedTasks, Task{
			ID:          updatedTask.ID,
			Description: updatedTask.Description,
			State:       state,
		})
	}

	return result, nil
}
