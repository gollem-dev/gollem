package reflexion

import (
	"context"
	"strings"

	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/goerr/v2"
)

// generateReflection generates a self-reflection from a failed trial.
// It uses the LLM to analyze what went wrong and how to improve.
func generateReflection(ctx context.Context, client gollem.LLMClient, trajectory *Trajectory, evaluation *EvaluationResult, memories []memoryEntry) (string, error) {
	prompt := buildReflectionPrompt(trajectory, evaluation, memories)

	// Create temporary session for reflection
	session, err := client.NewSession(ctx)
	if err != nil {
		return "", goerr.Wrap(err, "failed to create session for reflection")
	}

	resp, err := session.Generate(ctx, []gollem.Input{gollem.Text(prompt)})
	if err != nil {
		return "", goerr.Wrap(err, "failed to generate reflection")
	}

	return strings.Join(resp.Texts, "\n"), nil
}
