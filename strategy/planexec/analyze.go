package planexec

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/google/uuid"

	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/goerr/v2"
)

// GeneratePlan analyzes user input and creates an execution plan
// This can be called independently before creating an agent
// tools parameter specifies available tools for the agent to use during planning
// systemPrompt and history provide context that will be embedded into the plan
func GeneratePlan(ctx context.Context, client gollem.LLMClient, inputs []gollem.Input, tools []gollem.Tool, systemPrompt string, history *gollem.History) (*Plan, error) {
	if client == nil {
		return nil, goerr.New("client is required")
	}
	if len(inputs) == 0 {
		return nil, goerr.New("inputs are required")
	}

	return generatePlanInternal(ctx, client, inputs, tools, nil, systemPrompt, history)
}

// generatePlanInternal analyzes user input and creates a plan using LLM
// It uses system prompt and history to embed necessary context into the Plan's goal
// This is an internal analysis process - the conversation history is not preserved
func generatePlanInternal(ctx context.Context, client gollem.LLMClient, inputs []gollem.Input, tools []gollem.Tool, middleware []gollem.ContentBlockMiddleware, systemPrompt string, history *gollem.History) (*Plan, error) {
	// Create a new session with JSON content type
	// NOTE: Do NOT pass tools to planning session.
	// When tools are provided, some LLM providers (like Gemini) prioritize function calls
	// over JSON text responses, which breaks the planning phase that requires JSON output.
	sessionOpts := []gollem.SessionOption{
		gollem.WithSessionContentType(gollem.ContentTypeJSON),
	}

	// Add system prompt if provided
	// The system prompt helps the planner understand domain-specific constraints
	// which should be embedded into the Plan's `context_summary` and `constraints` fields.
	if systemPrompt != "" {
		sessionOpts = append(sessionOpts, gollem.WithSessionSystemPrompt(systemPrompt))
	}

	// Add history if provided
	// The history provides conversation context that should be considered
	// when creating the plan and embedded into the `context_summary` field.
	if history != nil {
		sessionOpts = append(sessionOpts, gollem.WithSessionHistory(history))
	}

	for _, mw := range middleware {
		sessionOpts = append(sessionOpts, gollem.WithSessionContentBlockMiddleware(mw))
	}

	session, err := client.NewSession(ctx, sessionOpts...)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to create session")
	}

	// Build planning prompt
	planPrompt := buildPlanPrompt(ctx, inputs, tools)

	// Generate plan using LLM
	response, err := session.Generate(ctx, planPrompt)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to generate plan")
	}

	// Parse the response to extract plan
	plan, err := parsePlanFromResponse(ctx, response)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to parse plan from response")
	}

	// Extract user's original question from inputs
	// This is used in the final conclusion to provide a direct answer to the user
	// Combine all text inputs to match the behavior of buildPlanPrompt
	var userTexts []string
	for _, input := range inputs {
		if text, ok := input.(gollem.Text); ok {
			userTexts = append(userTexts, string(text))
		}
	}
	if len(userTexts) > 0 {
		plan.UserQuestion = strings.Join(userTexts, " ")
	}

	return plan, nil
}

// parsePlanFromResponse extracts plan from LLM response
func parsePlanFromResponse(ctx context.Context, response *gollem.Response) (*Plan, error) {
	if response == nil {
		return nil, goerr.New("response is nil")
	}
	if len(response.Texts) == 0 {
		return nil, goerr.New("empty response from LLM")
	}

	// Parse JSON response directly (WithSessionContentType ensures JSON format)
	var planResponse struct {
		NeedsPlan      bool   `json:"needs_plan"`
		DirectResponse string `json:"direct_response"`
		UserIntent     string `json:"user_intent"`
		Goal           string `json:"goal"`
		ContextSummary string `json:"context_summary"`
		Constraints    string `json:"constraints"`
		Tasks          []struct {
			Description string `json:"description"`
		} `json:"tasks"`
	}

	if err := json.Unmarshal([]byte(response.Texts[0]), &planResponse); err != nil {
		return nil, goerr.Wrap(err, "failed to parse plan response as JSON")
	}

	// Create plan based on response
	if !planResponse.NeedsPlan {
		return &Plan{
			DirectResponse: planResponse.DirectResponse,
			Tasks:          []Task{},
		}, nil
	}

	// Convert to Plan with Tasks
	plan := &Plan{
		UserIntent:     planResponse.UserIntent,
		Goal:           planResponse.Goal,
		ContextSummary: planResponse.ContextSummary,
		Constraints:    planResponse.Constraints,
		Tasks:          make([]Task, len(planResponse.Tasks)),
	}

	for i, t := range planResponse.Tasks {
		plan.Tasks[i] = Task{
			ID:          uuid.New().String(),
			Description: t.Description,
			State:       TaskStatePending,
		}
	}

	return plan, nil
}
