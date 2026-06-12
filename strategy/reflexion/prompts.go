package reflexion

import (
	"fmt"
	"strings"

	"github.com/gollem-dev/gollem"
)

// buildEvaluationPrompt creates a prompt for evaluating whether a trial succeeded.
func buildEvaluationPrompt(trajectory *Trajectory) string {
	return fmt.Sprintf(`Evaluate whether the agent successfully completed the task.

Original task:
%s

Agent's execution history:
%s

Agent's final response:
%s

Did the agent successfully complete the task?
Respond with "SUCCESS" or "FAILURE" followed by a brief explanation.`,
		formatUserInputs(trajectory.UserInputs),
		formatHistory(trajectory.History),
		strings.Join(trajectory.FinalResponse, "\n"))
}

// buildReflectionPrompt creates a prompt for generating self-reflection after a failed trial.
// It includes 2-shot examples from the Reflexion paper (AlfWorld and HotPotQA).
func buildReflectionPrompt(trajectory *Trajectory, evaluation *EvaluationResult, memories []memoryEntry) string {
	prompt := `You are a reflective AI assistant. Analyze why you failed and provide specific guidance for improvement.

[Few-shot example 1: AlfWorld]
Task: Examine a mug with a desklamp
Execution: Found mug first, then looked for desklamp, but couldn't complete task
Reflection: In this environment, my plan was to find a mug then find and use a desklamp. However, the task says to examine the mug with the desklamp. I should have looked for the desklamp first, then looked for the mug. I noticed that the desklamp was found on desk 1. In the next trial, I will go to desk 1, find the lamp, then look for the mug and examine it with the desklamp.

[Few-shot example 2: HotPotQA]
Task: What role was the actor best known for in the TV show?
Execution: Searched for show title "'Allo 'Allo!" but got no results
Reflection: I searched the wrong title for the show, 'Allo 'Allo!', which resulted in no results. I should have searched the show's main character, Gorden Kaye, to find the role he was best known for.

Now analyze your own attempt:

Task: %s
Your execution: %s
Final response: %s
Evaluation: FAILURE - %s

Provide a concise reflection (100-300 words) covering:
1. What went wrong?
2. Why did it fail?
3. How should you improve in the next trial?`

	return fmt.Sprintf(prompt,
		formatUserInputs(trajectory.UserInputs),
		formatHistory(trajectory.History),
		strings.Join(trajectory.FinalResponse, "\n"),
		evaluation.Feedback)
}

// buildMemoryPrompt creates a prompt from episodic memory.
// It formats past reflections to help the agent learn from previous attempts.
func buildMemoryPrompt(memories []memoryEntry) gollem.Input {
	if len(memories) == 0 {
		return gollem.Text("")
	}

	var lines []string
	lines = append(lines, "You have attempted this task before. Here are your previous reflections:\n")

	for _, mem := range memories {
		lines = append(lines, fmt.Sprintf("Trial %d reflection:", mem.trialNum))
		lines = append(lines, mem.reflection)
		lines = append(lines, "")
	}

	lines = append(lines, "Learn from these past attempts to improve your strategy.")

	return gollem.Text(strings.Join(lines, "\n"))
}
