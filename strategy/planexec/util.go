package planexec

import (
	"context"
	"fmt"
	"strings"

	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/goerr/v2"
)

// getNextPendingTask returns the next task that needs to be executed
func getNextPendingTask(_ context.Context, plan *Plan) *Task {
	if plan == nil {
		return nil
	}

	for i := range plan.Tasks {
		if plan.Tasks[i].State == TaskStatePending {
			return &plan.Tasks[i]
		}
	}

	return nil
}

// allTasksCompleted checks if all tasks in the plan are completed or skipped
func allTasksCompleted(ctx context.Context, plan *Plan) bool {
	if plan == nil || len(plan.Tasks) == 0 {
		return true
	}

	for _, task := range plan.Tasks {
		// Tasks that are completed or skipped are considered "done"
		if task.State != TaskStateCompleted && task.State != TaskStateSkipped {
			return false
		}
	}

	return true
}

// getFinalConclusion asks LLM to generate final conclusion based on completed tasks
// Returns ExecuteResponse with texts and session history
func getFinalConclusion(ctx context.Context, client gollem.LLMClient, plan *Plan, middleware []gollem.ContentBlockMiddleware, systemPrompt string) (*gollem.ExecuteResponse, error) {
	if plan == nil {
		return &gollem.ExecuteResponse{
			Texts: []string{"No plan was executed."},
		}, nil
	}

	// If it was a direct response (no tasks), return it
	if len(plan.Tasks) == 0 && plan.DirectResponse != "" {
		return &gollem.ExecuteResponse{
			Texts: []string{plan.DirectResponse},
		}, nil
	}

	// Build summary of completed tasks
	var taskSummaries []string
	for _, task := range plan.Tasks {
		if task.State == TaskStateCompleted {
			summary := fmt.Sprintf("- %s", task.Description)
			if task.Result != "" {
				summary += fmt.Sprintf("\n  Result: %s", task.Result)
			}
			taskSummaries = append(taskSummaries, summary)
		}
	}

	// Create conclusion prompt using template
	conclusionPrompt := buildConclusionPrompt(plan, taskSummaries)

	// Create new session for conclusion
	sessionOpts := []gollem.SessionOption{}
	if systemPrompt != "" {
		sessionOpts = append(sessionOpts, gollem.WithSessionSystemPrompt(systemPrompt))
	}
	for _, mw := range middleware {
		sessionOpts = append(sessionOpts, gollem.WithSessionContentBlockMiddleware(mw))
	}

	session, err := client.NewSession(ctx, sessionOpts...)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to create session for conclusion")
	}

	// Generate conclusion
	response, err := session.Generate(ctx, []gollem.Input{gollem.Text(conclusionPrompt)})
	if err != nil {
		return nil, goerr.Wrap(err, "failed to generate conclusion")
	}

	// Return only the texts - the main session will automatically add them to history
	// No need to include AdditionalHistory as this is the final response, not an internal analysis
	return &gollem.ExecuteResponse{
		Texts: response.Texts,
	}, nil
}

// generateFinalResponse creates the final response from the completed plan (without LLM call)
func generateFinalResponse(ctx context.Context, plan *Plan) *gollem.ExecuteResponse {
	if plan == nil {
		return &gollem.ExecuteResponse{
			Texts: []string{"No plan was executed."},
		}
	}

	// If it was a direct response (no tasks), return it
	if len(plan.Tasks) == 0 && plan.DirectResponse != "" {
		return &gollem.ExecuteResponse{
			Texts: []string{plan.DirectResponse},
		}
	}

	// Build summary of completed tasks
	var results []string

	// Add goal if present
	if plan.Goal != "" {
		results = append(results, fmt.Sprintf("Goal: %s\n", plan.Goal))
	}

	// Add completed tasks and their results
	results = append(results, "Completed Tasks:")
	for i, task := range plan.Tasks {
		if task.State == TaskStateCompleted {
			results = append(results, fmt.Sprintf("%d. %s", i+1, task.Description))
			if task.Result != "" {
				// Indent result for readability
				resultLines := strings.Split(task.Result, "\n")
				for _, line := range resultLines {
					if line != "" {
						results = append(results, fmt.Sprintf("   %s", line))
					}
				}
			}
		}
	}

	// Join all results with newlines
	finalText := strings.Join(results, "\n")

	return &gollem.ExecuteResponse{
		Texts: []string{finalText},
	}
}
