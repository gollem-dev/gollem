package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"sync"

	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/goerr/v2"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const (
	// DefaultClientName is the default name for MCP client
	DefaultClientName = "gollem"
	// DefaultClientVersion is the default version for MCP client
	// Empty string means no specific version is advertised
	DefaultClientVersion = ""
)

// Client is the MCP client that allows to communicate with MCP server.
type Client struct {
	// Official SDK client
	mcpClient *mcp.Client
	session   *mcp.ClientSession

	// Configuration
	name    string
	version string

	// Transport related
	transport mcp.Transport
	cmd       *exec.Cmd // For stdio transport
	baseURL   string    // For StreamableHTTP transport

	// Options
	envVars    []string
	headers    map[string]string
	httpClient *http.Client // For StreamableHTTP transport

	// Connection management
	initMutex sync.Mutex
}

// Specs implements gollem.ToolSet interface
func (c *Client) Specs(ctx context.Context) ([]gollem.ToolSpec, error) {
	tools, err := c.listTools(ctx)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to list tools")
	}

	specs := make([]gollem.ToolSpec, len(tools))
	for i, tool := range tools {
		param, err := convertToolToSpec(tool)
		if err != nil {
			return nil, goerr.Wrap(err,
				"failed to convert tool to spec",
				goerr.V("tool.name", tool.Name),
			)
		}

		specs[i] = param
	}

	return specs, nil
}

// Run implements gollem.ToolSet interface
func (c *Client) Run(ctx context.Context, name string, args map[string]any) (map[string]any, error) {
	resp, err := c.callTool(ctx, name, args)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to call tool")
	}

	return convertContentToMap(resp.Content), nil
}

// StdioOption is the option for the MCP client for local MCP server via Stdio.
type StdioOption func(*Client)

// WithEnvVars sets the environment variables for the MCP client.
func WithEnvVars(envVars []string) StdioOption {
	return func(m *Client) {
		m.envVars = envVars
	}
}

// WithStdioClientInfo sets the client name and version for the MCP client.
func WithStdioClientInfo(name, version string) StdioOption {
	return func(m *Client) {
		m.name = name
		m.version = version
	}
}

// NewStdio creates a new MCP client for local MCP executable server via stdio.
func NewStdio(ctx context.Context, path string, args []string, options ...StdioOption) (*Client, error) {
	client := &Client{
		name:    DefaultClientName,
		version: DefaultClientVersion,
	}
	for _, option := range options {
		option(client)
	}

	// Create command with environment variables inheriting from the current process
	cmd := exec.Command(path, args...)
	cmd.Env = append(os.Environ(), client.envVars...)

	// Create transport
	transport := &mcp.StdioTransport{}
	client.transport = transport

	if err := client.init(ctx, cmd); err != nil {
		return nil, goerr.Wrap(err, "failed to initialize MCP client")
	}

	return client, nil
}

// NewSSE creates a new MCP client for remote MCP server via SSE.
func NewSSE(ctx context.Context, baseURL string, options ...SSEOption) (*Client, error) {
	client := &Client{
		name:       DefaultClientName,
		version:    DefaultClientVersion,
		headers:    make(map[string]string),
		baseURL:    baseURL,
		httpClient: http.DefaultClient,
	}
	for _, option := range options {
		option(client)
	}

	// Initialize the client and connect
	if err := client.initSSE(ctx); err != nil {
		return nil, goerr.Wrap(err, "failed to initialize SSE client")
	}

	return client, nil
}

// SSEOption is the option for the MCP client for remote MCP server via SSE.
type SSEOption func(*Client)

// WithSSEHeaders sets the headers for the MCP client. It replaces the existing headers setting.
func WithSSEHeaders(headers map[string]string) SSEOption {
	return func(m *Client) {
		m.headers = headers
	}
}

// WithSSEClient sets the HTTP client for the MCP client.
func WithSSEClient(client *http.Client) SSEOption {
	return func(m *Client) {
		m.httpClient = client
	}
}

// WithSSEClientInfo sets the client name and version for the MCP client.
func WithSSEClientInfo(name, version string) SSEOption {
	return func(m *Client) {
		m.name = name
		m.version = version
	}
}

// StreamableHTTPOption is the option for the MCP client for remote MCP server via Streamable HTTP.
type StreamableHTTPOption func(*Client)

// WithStreamableHTTPHeaders sets the headers for the MCP client. It replaces the existing headers setting.
func WithStreamableHTTPHeaders(headers map[string]string) StreamableHTTPOption {
	return func(m *Client) {
		m.headers = headers
	}
}

// WithStreamableHTTPClient sets the HTTP client for the MCP client.
func WithStreamableHTTPClient(client *http.Client) StreamableHTTPOption {
	return func(m *Client) {
		m.httpClient = client
	}
}

// WithStreamableHTTPClientInfo sets the client name and version for the MCP client.
func WithStreamableHTTPClientInfo(name, version string) StreamableHTTPOption {
	return func(m *Client) {
		m.name = name
		m.version = version
	}
}

// NewStreamableHTTP creates a new MCP client for remote MCP server via Streamable HTTP.
func NewStreamableHTTP(ctx context.Context, baseURL string, options ...StreamableHTTPOption) (*Client, error) {
	client := &Client{
		name:       DefaultClientName,
		version:    DefaultClientVersion,
		headers:    make(map[string]string),
		baseURL:    baseURL,
		httpClient: http.DefaultClient,
	}
	for _, option := range options {
		option(client)
	}

	// Initialize the client and connect
	if err := client.initStreamableHTTP(ctx); err != nil {
		return nil, goerr.Wrap(err, "failed to initialize StreamableHTTP client")
	}

	return client, nil
}

func (c *Client) init(ctx context.Context, cmd *exec.Cmd) error {
	c.initMutex.Lock()
	defer c.initMutex.Unlock()

	if c.session != nil {
		return nil
	}

	// Create client with official SDK using Implementation struct
	impl := &mcp.Implementation{
		Name:    c.name,
		Version: c.version,
	}
	c.mcpClient = mcp.NewClient(impl, nil)

	// Connect using stdio transport with command
	if cmd != nil {
		transport := &mcp.CommandTransport{
			Command: cmd,
		}
		session, err := c.mcpClient.Connect(ctx, transport, nil)
		if err != nil {
			return goerr.Wrap(err, "failed to connect to MCP server")
		}
		c.session = session
		c.cmd = cmd
	}

	return nil
}

func (c *Client) initStreamableHTTP(ctx context.Context) error {
	c.initMutex.Lock()
	defer c.initMutex.Unlock()

	if c.session != nil {
		return nil
	}

	// Create client with official SDK using Implementation struct
	impl := &mcp.Implementation{
		Name:    c.name,
		Version: c.version,
	}
	c.mcpClient = mcp.NewClient(impl, nil)

	// Create StreamableHTTP transport
	transport := &mcp.StreamableClientTransport{
		Endpoint:   c.baseURL,
		HTTPClient: c.httpClient,
	}

	// Connect using StreamableHTTP transport
	session, err := c.mcpClient.Connect(ctx, transport, nil)
	if err != nil {
		return goerr.Wrap(err, "failed to connect to StreamableHTTP MCP server")
	}
	c.session = session
	c.transport = transport

	return nil
}

func (c *Client) initSSE(ctx context.Context) error {
	c.initMutex.Lock()
	defer c.initMutex.Unlock()

	if c.session != nil {
		return nil
	}

	// Create client with official SDK using Implementation struct
	impl := &mcp.Implementation{
		Name:    c.name,
		Version: c.version,
	}
	c.mcpClient = mcp.NewClient(impl, nil)

	// Create SSE transport
	transport := &mcp.SSEClientTransport{
		Endpoint:   c.baseURL,
		HTTPClient: c.httpClient,
	}

	// Connect using SSE transport
	session, err := c.mcpClient.Connect(ctx, transport, nil)
	if err != nil {
		return goerr.Wrap(err, "failed to connect to SSE MCP server")
	}
	c.session = session
	c.transport = transport

	return nil
}

func (c *Client) listTools(ctx context.Context) ([]*mcp.Tool, error) {
	if c.session == nil {
		return nil, goerr.New("session not initialized")
	}

	resp, err := c.session.ListTools(ctx, &mcp.ListToolsParams{})
	if err != nil {
		return nil, goerr.Wrap(err, "failed to list tools")
	}

	return resp.Tools, nil
}

func (c *Client) callTool(ctx context.Context, name string, args map[string]any) (*mcp.CallToolResult, error) {
	if c.session == nil {
		return nil, goerr.New("session not initialized")
	}

	params := &mcp.CallToolParams{
		Name:      name,
		Arguments: args,
	}

	resp, err := c.session.CallTool(ctx, params)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to call tool")
	}

	return resp, nil
}

func (c *Client) Close() error {
	if c.session != nil {
		if err := c.session.Close(); err != nil {
			return goerr.Wrap(err, "failed to close MCP session")
		}
	}

	// Clean up stdio command process if it exists
	if c.cmd != nil && c.cmd.Process != nil {
		if err := c.cmd.Process.Kill(); err != nil {
			return goerr.Wrap(err, "failed to kill MCP server process")
		}
	}

	return nil
}

// convertToolToSpec converts MCP Tool to gollem.ToolSpec
func convertToolToSpec(tool *mcp.Tool) (gollem.ToolSpec, error) {
	spec := gollem.ToolSpec{
		Name:        tool.Name,
		Description: tool.Description,
		Parameters:  make(map[string]*gollem.Parameter),
	}

	// Convert input schema if available
	if tool.InputSchema != nil {
		param, err := convertInputSchemaToParameter(tool.InputSchema)
		if err != nil {
			return spec, goerr.Wrap(err, "failed to convert input schema")
		}
		spec.Parameters = param.Properties
	}

	return spec, nil
}

// convertInputSchemaToParameter converts MCP input schema to gollem Parameter
func convertInputSchemaToParameter(schema any) (*gollem.Parameter, error) {
	// Convert schema to JSON and back to map for processing
	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to marshal schema")
	}

	var schemaMap map[string]any
	if err := json.Unmarshal(schemaBytes, &schemaMap); err != nil {
		return nil, goerr.Wrap(err, "failed to unmarshal schema")
	}

	param := &gollem.Parameter{
		Type:       gollem.TypeObject,
		Properties: make(map[string]*gollem.Parameter),
	}

	// Extract properties
	if props, ok := schemaMap["properties"].(map[string]any); ok {
		for name, propSchema := range props {
			propParam, err := convertSchemaProperty(propSchema)
			if err != nil {
				return nil, goerr.Wrap(err, "failed to convert property", goerr.V("property", name))
			}
			param.Properties[name] = propParam
		}
	}

	// Extract required fields and set Required bool on each property
	if required, ok := schemaMap["required"].([]any); ok {
		for _, req := range required {
			if reqStr, ok := req.(string); ok {
				if prop, exists := param.Properties[reqStr]; exists {
					prop.Required = true
				}
			}
		}
	}

	return param, nil
}

// convertSchemaProperty converts a single schema property to gollem Parameter
func convertSchemaProperty(propSchema any) (*gollem.Parameter, error) {
	propBytes, err := json.Marshal(propSchema)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to marshal property schema")
	}

	var propMap map[string]any
	if err := json.Unmarshal(propBytes, &propMap); err != nil {
		return nil, goerr.Wrap(err, "failed to unmarshal property schema")
	}

	param := &gollem.Parameter{}

	// Type
	if typeVal, ok := propMap["type"].(string); ok {
		param.Type = gollem.ParameterType(typeVal)
	}

	// Description
	if desc, ok := propMap["description"].(string); ok {
		param.Description = desc
	}

	// Title
	if title, ok := propMap["title"].(string); ok {
		param.Title = title
	}

	// Default value
	if defaultVal, ok := propMap["default"]; ok {
		param.Default = defaultVal
	}

	// Handle enum
	if enumVal, ok := propMap["enum"].([]any); ok {
		for _, e := range enumVal {
			param.Enum = append(param.Enum, fmt.Sprintf("%v", e))
		}
	}

	// Handle object type - recursive processing of properties
	if param.Type == gollem.TypeObject {
		param.Properties = make(map[string]*gollem.Parameter)

		// Extract and recursively process properties
		if props, ok := propMap["properties"].(map[string]any); ok {
			for name, propSchema := range props {
				nestedParam, err := convertSchemaProperty(propSchema)
				if err != nil {
					return nil, goerr.Wrap(err, "failed to convert nested property", goerr.V("property", name))
				}
				param.Properties[name] = nestedParam
			}
		}

		// Extract required fields and set Required bool on each property
		if required, ok := propMap["required"].([]any); ok {
			for _, req := range required {
				if reqStr, ok := req.(string); ok {
					if prop, exists := param.Properties[reqStr]; exists {
						prop.Required = true
					}
				}
			}
		}
	}

	// Handle array type - recursive processing of items
	if param.Type == gollem.TypeArray {
		if items, ok := propMap["items"]; ok {
			itemParam, err := convertSchemaProperty(items)
			if err != nil {
				return nil, goerr.Wrap(err, "failed to convert array items")
			}
			param.Items = itemParam
		}

		// Array constraints
		if minItems, ok := propMap["minItems"].(float64); ok {
			val := int(minItems)
			param.MinItems = &val
		}
		if maxItems, ok := propMap["maxItems"].(float64); ok {
			val := int(maxItems)
			param.MaxItems = &val
		}
	}

	// Number constraints
	if param.Type == gollem.TypeNumber || param.Type == gollem.TypeInteger {
		if minimum, ok := propMap["minimum"].(float64); ok {
			param.Minimum = &minimum
		}
		if maximum, ok := propMap["maximum"].(float64); ok {
			param.Maximum = &maximum
		}
	}

	// String constraints
	if param.Type == gollem.TypeString {
		if minLength, ok := propMap["minLength"].(float64); ok {
			val := int(minLength)
			param.MinLength = &val
		}
		if maxLength, ok := propMap["maxLength"].(float64); ok {
			val := int(maxLength)
			param.MaxLength = &val
		}
		if pattern, ok := propMap["pattern"].(string); ok {
			param.Pattern = pattern
		}
	}

	return param, nil
}

// convertContentToMap converts MCP Content to map[string]any
func convertContentToMap(contents []mcp.Content) map[string]any {
	if len(contents) == 0 {
		return nil
	}

	if len(contents) == 1 {
		if textContent, ok := contents[0].(*mcp.TextContent); ok {
			var v any
			if err := json.Unmarshal([]byte(textContent.Text), &v); err == nil {
				if mapData, ok := v.(map[string]any); ok {
					return mapData
				}
			}
			return map[string]any{
				"result": textContent.Text,
			}
		}
		return nil
	}

	result := map[string]any{}
	for i, c := range contents {
		if textContent, ok := c.(*mcp.TextContent); ok {
			result[fmt.Sprintf("content_%d", i+1)] = textContent.Text
		}
	}
	return result
}
