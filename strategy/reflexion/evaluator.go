package reflexion

import (
	"context"
	"strings"

	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/goerr/v2"
)

// NewLLMEvaluator creates a new LLM-based evaluator.
// It asks the LLM to determine if the task was successfully completed
// based on the original task, execution history, and final response.
func NewLLMEvaluator(client gollem.LLMClient) Evaluator {
	return func(ctx context.Context, trajectory *Trajectory) (*EvaluationResult, error) {
		prompt := buildEvaluationPrompt(trajectory)

		// Create temporary session for evaluation
		session, err := client.NewSession(ctx)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to create session for evaluation")
		}

		resp, err := session.Generate(ctx, []gollem.Input{gollem.Text(prompt)})
		if err != nil {
			return nil, goerr.Wrap(err, "failed to generate evaluation")
		}

		// Parse response for SUCCESS/FAILURE
		result := parseEvaluationResponse(resp)

		return result, nil
	}
}

// parseEvaluationResponse parses the LLM's evaluation response.
// It looks for "SUCCESS" or "FAILURE" keywords in the response.
func parseEvaluationResponse(resp *gollem.Response) *EvaluationResult {
	text := strings.Join(resp.Texts, "\n")
	textUpper := strings.ToUpper(text)

	success := strings.Contains(textUpper, "SUCCESS")

	return &EvaluationResult{
		Success:  success,
		Feedback: text,
	}
}
