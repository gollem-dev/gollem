package gollem

import (
	"bytes"
	"context"
	"strings"
	"text/template"

	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/goerr/v2"
)

// SubAgent represents an agent that can be invoked as a tool by parent agent.
// SubAgent implements the Tool interface, allowing it to be added to an agent's tool list.
type SubAgent struct {
	name         string
	description  string
	agentFactory func() (*Agent, error)

	// Template mode fields (nil when using default query-only mode)
	parsedTemplate *template.Template    // Parsed template (cached)
	parameters     map[string]*Parameter // Parameter schema for LLM

	// Middleware for processing arguments before template rendering
	middleware func(SubAgentHandler) SubAgentHandler

	// Options to apply to child agents created by agentFactory
	subAgentOptions []Option
}

// SubAgentOption is the type for options when creating a SubAgent.
type SubAgentOption func(*SubAgent)

// SubAgentResult holds the execution result of a SubAgent handler.
// It contains both the data returned to the caller and the session
// for post-execution processing by middleware.
type SubAgentResult struct {
	Data    map[string]any // Result data to be returned to the parent agent
	Session Session        // Session used during execution (read-only access for middleware)
}

// SubAgentHandler is a function that processes arguments for SubAgent execution.
// It receives the context and current arguments, and returns SubAgentResult containing
// both the result data and the session for middleware post-processing.
type SubAgentHandler func(ctx context.Context, args map[string]any) (SubAgentResult, error)

// PromptTemplate holds the parsed template and parameter schema for template mode.
// Use NewPromptTemplate to create an instance with proper error handling.
type PromptTemplate struct {
	parsedTemplate *template.Template
	parameters     map[string]*Parameter
}

// NewPromptTemplate creates a new PromptTemplate with the given template and parameters.
// tmpl: Go text/template string to generate the prompt
// params: Parameter schema for LLM (key is parameter name)
//
// The template uses missingkey=zero option, so missing variables are replaced with
// their zero value (empty string for strings). This allows conditional checks like
// {{if .optional}}...{{end}} for optional parameters.
//
// Returns an error if the template string is invalid.
func NewPromptTemplate(tmpl string, params map[string]*Parameter) (*PromptTemplate, error) {
	parsed, err := template.New("prompt").Option("missingkey=zero").Parse(tmpl)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to parse template")
	}
	return &PromptTemplate{
		parsedTemplate: parsed,
		parameters:     params,
	}, nil
}

// DefaultPromptTemplate returns the default prompt template that accepts a single "query" parameter.
// This is equivalent to the default behavior when no prompt template is specified.
func DefaultPromptTemplate() *PromptTemplate {
	parsed := template.Must(template.New("prompt").Option("missingkey=zero").Parse("{{.query}}"))
	return &PromptTemplate{
		parsedTemplate: parsed,
		parameters: map[string]*Parameter{
			"query": {
				Type:        TypeString,
				Description: "Natural language query to send to the subagent",
				Required:    true,
			},
		},
	}
}

// Render renders the template with the given arguments and returns the resulting prompt string.
// This method is useful for testing templates independently from the SubAgent.
// Returns an error if validation fails (missing required params, type mismatch, constraint violations).
// Missing optional parameters are replaced with their zero value (empty string for strings).
// Arguments not defined in parameters (e.g., injected by middleware) are also available in the template.
func (p *PromptTemplate) Render(args map[string]any) (string, error) {
	// Validate all defined parameters and build data map
	data := make(map[string]any, len(args))
	for name, param := range p.parameters {
		value := args[name]
		if err := param.ValidateValue(name, value); err != nil {
			return "", err
		}
		if value != nil {
			data[name] = value
		} else {
			// Set zero value for missing optional parameters to avoid <no value>
			data[name] = zeroValue(param.Type)
		}
	}

	// Add any additional args that are not in parameters (e.g., from middleware)
	for name, value := range args {
		if _, exists := p.parameters[name]; !exists {
			data[name] = value
		}
	}

	var buf bytes.Buffer
	if err := p.parsedTemplate.Execute(&buf, data); err != nil {
		return "", goerr.Wrap(err, "failed to execute template")
	}
	return buf.String(), nil
}

// zeroValue returns the zero value for a given parameter type.
func zeroValue(t ParameterType) any {
	switch t {
	case TypeString:
		return ""
	case TypeNumber, TypeInteger:
		return 0
	case TypeBoolean:
		return false
	case TypeArray:
		return []any{}
	case TypeObject:
		return map[string]any{}
	default:
		return ""
	}
}

// Parameters returns the parameter schema for this template.
// This is useful for inspecting the expected parameters.
func (p *PromptTemplate) Parameters() map[string]*Parameter {
	return p.parameters
}

// WithPromptTemplate sets a pre-configured prompt template for the subagent.
// This option replaces the default "query" parameter behavior.
// When this option is used, the template is rendered with the provided parameters
// and the result is passed to agent.Execute().
func WithPromptTemplate(prompt *PromptTemplate) SubAgentOption {
	return func(s *SubAgent) {
		s.parsedTemplate = prompt.parsedTemplate
		s.parameters = prompt.parameters
	}
}

// WithSubAgentMiddleware sets a middleware that can modify arguments before template rendering.
// Multiple middlewares can be chained by calling this option multiple times.
// The middleware follows the same pattern as WithToolMiddleware and WithContentBlockMiddleware.
//
// The middleware can:
//   - Add context information (timestamps, user data, environment info)
//   - Modify or transform arguments from the LLM
//   - Perform logging or monitoring
//   - Validate or filter arguments
//
// Example:
//
//	gollem.WithSubAgentMiddleware(func(next gollem.SubAgentHandler) gollem.SubAgentHandler {
//	    return func(ctx context.Context, args map[string]any) (map[string]any, error) {
//	        // Add context before processing
//	        args["current_time"] = time.Now().Format(time.RFC3339)
//	        return next(ctx, args)
//	    }
//	})
func WithSubAgentMiddleware(middleware func(SubAgentHandler) SubAgentHandler) SubAgentOption {
	return func(s *SubAgent) {
		if s.middleware == nil {
			s.middleware = middleware
		} else {
			// Chain middlewares
			prev := s.middleware
			s.middleware = func(next SubAgentHandler) SubAgentHandler {
				return prev(middleware(next))
			}
		}
	}
}

// WithSubAgentOptions sets additional gollem.Option values to apply to child agents
// created by the factory function. These options are applied after the factory
// creates the agent but before Execute() is called.
//
// This enables the parent agent to inject cross-cutting concerns (such as
// tool middleware for budget tracking or monitoring) into sub-agent execution,
// without modifying each sub-agent's factory function.
//
// Multiple calls to this option are cumulative - options are appended.
func WithSubAgentOptions(opts ...Option) SubAgentOption {
	return func(s *SubAgent) {
		s.subAgentOptions = append(s.subAgentOptions, opts...)
	}
}

// NewSubAgent creates a new SubAgent that wraps a factory function for creating Agent instances.
// name: Tool name for the subagent (required, used by LLM to invoke)
// description: Description of what this subagent does (required, helps LLM decide when to use)
// agentFactory: A function that creates a new Agent instance (required). Returns error if creation fails.
//
// The factory function is called each time the SubAgent is invoked, ensuring that
// each execution has an independent session state. This prevents session state
// from being shared across multiple calls or Chat sessions.
func NewSubAgent(name, description string, agentFactory func() (*Agent, error), opts ...SubAgentOption) *SubAgent {
	s := &SubAgent{
		name:         name,
		description:  description,
		agentFactory: agentFactory,
	}

	for _, opt := range opts {
		opt(s)
	}

	return s
}

// Spec returns the ToolSpec for this SubAgent.
// In default mode, it returns a spec with a single "query" parameter.
// In template mode, it returns a spec with the custom parameters.
func (s *SubAgent) Spec() ToolSpec {
	var params map[string]*Parameter

	if s.parsedTemplate == nil {
		// Default mode: query only
		params = map[string]*Parameter{
			"query": {
				Type:        TypeString,
				Description: "Natural language query to send to the subagent",
				Required:    true,
			},
		}
	} else {
		// Template mode: use custom parameters
		params = s.parameters
	}

	return ToolSpec{
		Name:        s.name,
		Description: s.description,
		Parameters:  params,
	}
}

// Run executes the SubAgent with the given arguments.
// In default mode, it uses the default prompt template with a "query" parameter.
// In template mode, it renders the template with the arguments and passes the result to the agent.
// If middleware is set, it is applied to the arguments before template rendering.
func (s *SubAgent) Run(ctx context.Context, args map[string]any) (_ map[string]any, retErr error) {
	// Start sub-agent trace span
	if h := trace.HandlerFrom(ctx); h != nil {
		ctx = h.StartSubAgent(ctx, s.name)
		defer func() { h.EndSubAgent(ctx, retErr) }()
	}

	// Define the core handler that renders template and executes agent
	coreHandler := func(ctx context.Context, args map[string]any) (SubAgentResult, error) {
		var pt *PromptTemplate
		if s.parsedTemplate == nil {
			// Default mode: use default prompt template
			pt = DefaultPromptTemplate()
		} else {
			// Template mode: use custom template
			pt = &PromptTemplate{
				parsedTemplate: s.parsedTemplate,
				parameters:     s.parameters,
			}
		}

		prompt, err := pt.Render(args)
		if err != nil {
			return SubAgentResult{}, err
		}

		// Create a new agent instance for this execution
		agent, err := s.agentFactory()
		if err != nil {
			return SubAgentResult{}, goerr.Wrap(err, "failed to create agent from factory").Wrap(ErrSubAgentFactory)
		}
		if agent == nil {
			return SubAgentResult{}, goerr.New("agent factory returned nil").Wrap(ErrSubAgentFactory)
		}

		// Apply additional options to the child agent
		for _, opt := range s.subAgentOptions {
			opt(&agent.gollemConfig)
		}

		// Execute the child agent
		resp, err := agent.Execute(ctx, Text(prompt))
		if err != nil {
			return SubAgentResult{}, goerr.Wrap(err, "subagent execution failed")
		}

		// Build response text from ExecuteResponse
		var responseText string
		if resp != nil && len(resp.Texts) > 0 {
			responseText = strings.Join(resp.Texts, "\n")
		}

		// Return SubAgentResult with both data and session
		return SubAgentResult{
			Data: map[string]any{
				"response": responseText,
				"status":   "success",
			},
			Session: agent.Session(),
		}, nil
	}

	// Apply middleware if set
	handler := coreHandler
	if s.middleware != nil {
		handler = s.middleware(coreHandler)
	}

	// Execute and extract Data from result
	result, err := handler(ctx, args)
	if err != nil {
		return nil, err
	}

	// Return only Data to parent agent (Session is not exposed)
	return result.Data, nil
}
