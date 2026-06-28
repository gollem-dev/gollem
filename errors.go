package gollem

import (
	"errors"

	"github.com/m-mizutani/goerr/v2"
)

var (
	// ErrInvalidTool is returned when the tool validation of definition fails.
	ErrInvalidTool = errors.New("invalid tool specification")

	// ErrInvalidParameter is returned when the parameter validation of definition fails.
	ErrInvalidParameter = errors.New("invalid parameter")

	// ErrToolNameConflict is returned when the tool name is already used.
	ErrToolNameConflict = errors.New("tool name conflict")

	// ErrLoopLimitExceeded is returned when the session loop limit is exceeded. You can resume the session by calling the Prompt() method again.
	ErrLoopLimitExceeded = errors.New("loop limit exceeded")

	// ErrInvalidInputSchema is returned when the input schema from MCP is invalid or unsupported.
	ErrInvalidInputSchema = errors.New("invalid input schema")

	// ErrInvalidToolType is returned when a typed tool's In/Out type parameter cannot form a valid schema.
	ErrInvalidToolType = errors.New("invalid tool type")

	// ErrInvalidHistoryData is returned when the history data is invalid or unsupported.
	ErrInvalidHistoryData = errors.New("invalid history data")

	// ErrLLMTypeMismatch is returned when the LLM type is invalid or unsupported when loading history.
	ErrLLMTypeMismatch = errors.New("llm type mismatch")

	// ErrHistoryVersionMismatch is returned when the history version is invalid or unsupported.
	ErrHistoryVersionMismatch = errors.New("history version mismatch")

	// ErrExitConversation is returned when a tool signals that the conversation should be exited.
	// This error is treated as a successful completion of the conversation loop.
	ErrExitConversation = errors.New("exit conversation")

	// Plan mode specific errors

	// ErrPlanAlreadyExecuted is returned when trying to run an already executed plan
	ErrPlanAlreadyExecuted = errors.New("plan already executed")

	// ErrPlanNotInitialized is returned when plan is not properly initialized
	ErrPlanNotInitialized = errors.New("plan not properly initialized")

	// ErrPlanStepFailed is returned when a plan step fails during execution
	ErrPlanStepFailed = errors.New("plan step execution failed")

	// ErrTokenSizeExceeded is returned when the token size exceeds the limit
	ErrTokenSizeExceeded = errors.New("token size exceeded")

	// ErrFunctionCallFormat is returned when the function call format is invalid
	ErrFunctionCallFormat = errors.New("function call format error")

	// ErrProhibitedContent is returned when the content violates policy
	ErrProhibitedContent = errors.New("prohibited content")

	// ErrToolArgsValidation is returned when the tool arguments from LLM fail validation.
	// This is distinct from ErrInvalidParameter which is for spec definition validation.
	ErrToolArgsValidation = errors.New("tool arguments validation failed")

	// ErrSubAgentFactory is returned when the subagent factory fails to create an agent.
	ErrSubAgentFactory = errors.New("subagent factory failed")

	// ErrTagTokenExceeded is a tag for errors caused by token limit exceeded
	ErrTagTokenExceeded = goerr.NewTag("token_exceeded")
)
