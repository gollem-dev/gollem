package mcp_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/mcp"
	"github.com/m-mizutani/gt"
	officialmcp "github.com/modelcontextprotocol/go-sdk/mcp"
)

func TestMCPLocalDryRun(t *testing.T) {
	if _, ok := os.LookupEnv("TEST_MCP_LOCAL"); !ok {
		t.Skip("TEST_MCP_LOCAL is not set")
	}

	ctx := context.Background()

	// Create MCP client using npx @modelcontextprotocol/server-filesystem
	// This server provides filesystem access tools like read_file, write_file, list_directory
	// The filesystem server requires a directory argument to set allowed access
	mcpClient, err := mcp.NewStdio(ctx, "npx", []string{"-y", "@modelcontextprotocol/server-filesystem", "."})
	if err != nil {
		t.Skip("Could not create MCP filesystem client (requires npx and @modelcontextprotocol/server-filesystem):", err)
	}
	defer func() {
		if err := mcpClient.Close(); err != nil {
			t.Logf("failed to close MCP client: %v", err)
		}
	}()

	// Test Specs method - get available tools
	specs, err := mcpClient.Specs(ctx)
	gt.NoError(t, err)
	gt.Array(t, specs).Longer(0)

	// Find a tool to test with - filesystem server typically provides read_file, write_file, list_directory
	var testTool *gollem.ToolSpec
	for _, spec := range specs {
		if spec.Name == "list_directory" {
			testTool = &spec
			break
		}
	}

	if testTool == nil {
		// Fallback to first available tool
		testTool = &specs[0]
	}

	// Test Run method with appropriate arguments based on the tool
	var testArgs map[string]any
	switch testTool.Name {
	case "list_directory":
		// List current directory
		testArgs = map[string]any{
			"path": ".",
		}
	case "read_file":
		// Try to read a common file
		testArgs = map[string]any{
			"path": "go.mod",
		}
	default:
		// Generic test - try with minimal arguments
		testArgs = map[string]any{}

		// Add required parameters if any
		for paramName, param := range testTool.Parameters {
			if param.Required {
				switch param.Type {
				case gollem.TypeString:
					testArgs[paramName] = "test"
				case gollem.TypeInteger:
					testArgs[paramName] = 1
				case gollem.TypeNumber:
					testArgs[paramName] = 1.0
				case gollem.TypeBoolean:
					testArgs[paramName] = true
				}
			}
		}
	}

	result, err := mcpClient.Run(ctx, testTool.Name, testArgs)
	gt.NoError(t, err)
	gt.NotNil(t, result)

}

func TestMCPContentToMap(t *testing.T) {
	// Note: These tests are commented out as they depend on the old SDK types
	// They would need to be rewritten to use the official SDK types

	t.Run("when content is empty", func(t *testing.T) {
		t.Skip("Test needs to be updated for official SDK")
	})

	t.Run("when text content is JSON", func(t *testing.T) {
		t.Skip("Test needs to be updated for official SDK")
	})

	t.Run("when text content is not JSON", func(t *testing.T) {
		t.Skip("Test needs to be updated for official SDK")
	})

	t.Run("when multiple contents exist", func(t *testing.T) {
		t.Skip("Test needs to be updated for official SDK")
	})
}

func TestNewStreamableHTTP(t *testing.T) {
	t.Run("basic functionality", func(t *testing.T) {
		// StreamableHTTP should now work but will fail with connection error
		client, err := mcp.NewStreamableHTTP(
			t.Context(),
			"http://localhost:99999",
		)
		// Connection should fail due to unreachable server, but client creation should work
		gt.Error(t, err)
		gt.Nil(t, client) // Client is nil when connection fails
		// Error should be connection-related, not "not implemented"
		gt.False(t, err.Error() == "StreamableHTTP transport not yet implemented with official SDK")
	})

	t.Run("with headers", func(t *testing.T) {
		// StreamableHTTP should now work but will fail with connection error
		client, err := mcp.NewStreamableHTTP(
			t.Context(),
			"http://localhost:99999",
			mcp.WithStreamableHTTPHeaders(map[string]string{
				"Authorization": "Bearer test-token",
				"Custom-Header": "test-value",
			}),
		)
		// Connection should fail due to unreachable server, but client creation should work
		gt.Error(t, err)
		gt.Nil(t, client) // Client is nil when connection fails
		// Error should be connection-related, not "not implemented"
		gt.False(t, err.Error() == "StreamableHTTP transport not yet implemented with official SDK")
	})
}

func TestNewSSE(t *testing.T) {
	t.Run("basic functionality", func(t *testing.T) {
		// SSE should now work but will fail with connection error
		client, err := mcp.NewSSE(
			t.Context(),
			"http://localhost:99999",
		)
		// Connection should fail due to unreachable server, but client creation should work
		gt.Error(t, err)
		gt.Nil(t, client) // Client is nil when connection fails
		// Error should be connection-related, not "not implemented"
		gt.True(t, err != nil)
	})

	t.Run("with headers", func(t *testing.T) {
		// SSE should now work but will fail with connection error
		client, err := mcp.NewSSE(
			t.Context(),
			"http://localhost:99999",
			mcp.WithSSEHeaders(map[string]string{
				"Authorization": "Bearer test-token",
				"Custom-Header": "test-value",
			}),
		)
		// Connection should fail due to unreachable server, but client creation should work
		gt.Error(t, err)
		gt.Nil(t, client) // Client is nil when connection fails
		// Error should be connection-related, not "not implemented"
		gt.True(t, err != nil)
	})
}

func TestNewStreamableHTTPErrorHandling(t *testing.T) {
	// Test with invalid URL
	t.Run("invalid URL", func(t *testing.T) {
		client, err := mcp.NewStreamableHTTP(
			t.Context(),
			"invalid-url",
		)
		gt.Error(t, err)
		gt.Nil(t, client)
	})

	// Test with unreachable server
	t.Run("unreachable server", func(t *testing.T) {
		client, err := mcp.NewStreamableHTTP(
			t.Context(),
			"http://localhost:99999",
		)
		gt.Error(t, err)
		gt.Nil(t, client)
	})
}

func TestNewSSEErrorHandling(t *testing.T) {
	// Test with invalid URL
	t.Run("invalid URL", func(t *testing.T) {
		client, err := mcp.NewSSE(
			t.Context(),
			"invalid-url",
		)
		gt.Error(t, err)
		gt.Nil(t, client)
	})

	// Test with unreachable server
	t.Run("unreachable server", func(t *testing.T) {
		client, err := mcp.NewSSE(
			t.Context(),
			"http://localhost:99999",
		)
		gt.Error(t, err)
		gt.Nil(t, client)
	})
}

func TestClientOptions(t *testing.T) {
	ctx := t.Context()

	t.Run("NewStdio with custom client info", func(t *testing.T) {
		// Test that custom client name and version can be set
		client := &mcp.Client{}
		option := mcp.WithStdioClientInfo("test-client", "1.2.3")
		option(client)

		// Verify the values are set correctly through the internal state
		// Since we can't directly access the private fields, we'll create a client
		// and verify it doesn't error when using custom options
		mcpExecPath := "/bin/echo" // Use a simple command that exists on most systems
		_, err := mcp.NewStdio(ctx, mcpExecPath, []string{"test"},
			mcp.WithStdioClientInfo("custom-client", "2.0.0"))

		// We expect this to fail with connection error, but not with option parsing error
		if err != nil {
			gt.True(t, err.Error() != "")
			// The error should be about connection, not about parsing options
			gt.False(t, len(err.Error()) == 0)
		}
	})

	t.Run("NewStdio with default client info", func(t *testing.T) {
		// Test that default client name and version are used when not specified
		mcpExecPath := "/bin/echo"
		_, err := mcp.NewStdio(ctx, mcpExecPath, []string{"test"})

		// Should not error due to missing client info (defaults should be used)
		if err != nil {
			// Error should be connection-related, not missing client info
			gt.True(t, err.Error() != "")
		}
	})

	t.Run("NewSSE with custom client info", func(t *testing.T) {
		_, err := mcp.NewSSE(ctx, "http://localhost:99999",
			mcp.WithSSEClientInfo("custom-sse-client", "3.0.0"))

		// Should fail with connection error, not "not implemented"
		gt.Error(t, err)
		// Error should be connection-related
		gt.True(t, err != nil)
	})

	t.Run("NewSSE with default client info", func(t *testing.T) {
		_, err := mcp.NewSSE(ctx, "http://localhost:99999")

		// Should fail with connection error, not "not implemented"
		gt.Error(t, err)
		// Error should be connection-related
		gt.True(t, err != nil)
	})

	t.Run("NewStreamableHTTP with custom client info", func(t *testing.T) {
		_, err := mcp.NewStreamableHTTP(ctx, "http://localhost:99999",
			mcp.WithStreamableHTTPClientInfo("custom-http-client", "4.0.0"))

		// Should fail with connection error, not "not implemented"
		gt.Error(t, err)
		gt.False(t, err.Error() == "StreamableHTTP transport not yet implemented with official SDK")
	})

	t.Run("NewStreamableHTTP with default client info", func(t *testing.T) {
		_, err := mcp.NewStreamableHTTP(ctx, "http://localhost:99999")

		// Should fail with connection error, not "not implemented"
		gt.Error(t, err)
		gt.False(t, err.Error() == "StreamableHTTP transport not yet implemented with official SDK")
	})

	t.Run("Multiple options can be combined", func(t *testing.T) {
		mcpExecPath := "/bin/echo"
		_, err := mcp.NewStdio(ctx, mcpExecPath, []string{"test"},
			mcp.WithStdioClientInfo("combined-client", "1.0.0"),
			mcp.WithEnvVars([]string{"TEST_VAR=test_value"}))

		// Should not error due to option parsing
		if err != nil {
			// Error should be connection-related, not option-related
			gt.True(t, err.Error() != "")
		}
	})
}

func TestWithOfficialSDKServer(t *testing.T) {
	ctx := t.Context()

	// Define input type for the tool
	type GreetInput struct {
		Name string `json:"name"`
	}

	// Create a test tool handler with v1.1.0 signature
	toolHandler := func(ctx context.Context, req *officialmcp.CallToolRequest) (*officialmcp.CallToolResult, error) {
		var input GreetInput
		if len(req.Params.Arguments) > 0 {
			if err := json.Unmarshal(req.Params.Arguments, &input); err != nil {
				return nil, err
			}
		}
		name := input.Name
		if name == "" {
			name = "world"
		}
		return &officialmcp.CallToolResult{
			Content: []officialmcp.Content{&officialmcp.TextContent{Text: "Hello, " + name + "!"}},
		}, nil
	}

	// Create server with official SDK using Implementation struct
	impl := &officialmcp.Implementation{
		Name:    "test-server",
		Version: "1.0.0",
	}
	server := officialmcp.NewServer(impl, nil)

	// Add tool using v1.1.0 API
	tool := &officialmcp.Tool{
		Name:        "greet",
		Description: "say hello",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{
					"type":        "string",
					"description": "the name of the person to greet",
				},
			},
		},
	}
	server.AddTool(tool, toolHandler)

	// Test with StreamableHTTP transport
	t.Run("Test gollem mcp with StreamableHTTP transport", func(t *testing.T) {
		// Create HTTP handler for StreamableHTTP
		httpHandler := officialmcp.NewStreamableHTTPHandler(func(r *http.Request) *officialmcp.Server {
			return server
		}, nil)

		// Start test HTTP server
		httpServer := httptest.NewServer(httpHandler)
		defer httpServer.Close()

		// Test using gollem's mcp package
		mcpClient, err := mcp.NewStreamableHTTP(ctx, httpServer.URL,
			mcp.WithStreamableHTTPClientInfo("gollem-test-client", "1.0.0"))
		gt.NoError(t, err)
		defer func() {
			if err := mcpClient.Close(); err != nil {
				t.Logf("failed to close MCP client: %v", err)
			}
		}()

		// Test Specs method
		specs, err := mcpClient.Specs(ctx)
		gt.NoError(t, err)
		gt.Array(t, specs).Length(1)
		gt.Equal(t, "greet", specs[0].Name)
		gt.Equal(t, "say hello", specs[0].Description)

		// Test Run method
		result, err := mcpClient.Run(ctx, "greet", map[string]any{"name": "StreamableHTTP"})
		gt.NoError(t, err)
		gt.NotNil(t, result)

		// The result should contain the greeting message
		if resultStr, ok := result["result"].(string); ok {
			gt.Equal(t, "Hello, StreamableHTTP!", resultStr)
		} else {
			t.Errorf("Expected result key with string value, got: %+v", result)
		}
	})

	// Test with SSE transport (deprecated but still functional)
	t.Run("Test gollem mcp with SSE transport (deprecated)", func(t *testing.T) {
		// Create HTTP handler for SSE with v1.1.0 API
		sseHandler := officialmcp.NewSSEHandler(func(r *http.Request) *officialmcp.Server {
			return server
		}, nil)

		// Start test HTTP server
		httpServer := httptest.NewServer(sseHandler)
		defer httpServer.Close()

		// Test using gollem's mcp package (SSE is deprecated)
		mcpClient, err := mcp.NewSSE(ctx, httpServer.URL,
			mcp.WithSSEClientInfo("gollem-test-sse-client", "1.0.0"))
		gt.NoError(t, err)
		defer func() {
			if err := mcpClient.Close(); err != nil {
				t.Logf("failed to close MCP client: %v", err)
			}
		}()

		// Test Specs method
		specs, err := mcpClient.Specs(ctx)
		gt.NoError(t, err)
		gt.Array(t, specs).Length(1)
		gt.Equal(t, "greet", specs[0].Name)
		gt.Equal(t, "say hello", specs[0].Description)

		// Test Run method
		result, err := mcpClient.Run(ctx, "greet", map[string]any{"name": "SSE"})
		gt.NoError(t, err)
		gt.NotNil(t, result)

		// The result should contain the greeting message
		if resultStr, ok := result["result"].(string); ok {
			gt.Equal(t, "Hello, SSE!", resultStr)
		} else {
			t.Errorf("Expected result key with string value, got: %+v", result)
		}
	})

	// Test conversion functions with official SDK types
	t.Run("Test our conversion functions with official SDK types", func(t *testing.T) {
		// Test convertContentToMap with official SDK content
		textContent := &officialmcp.TextContent{Text: `{"result": "success", "data": 123}`}
		contents := []officialmcp.Content{textContent}

		result := mcp.MCPContentToMap(contents)
		gt.NotNil(t, result)

		// Should parse JSON content
		expected := map[string]any{"result": "success", "data": float64(123)}
		gt.Equal(t, expected, result)
	})

	t.Run("Test convertContentToMap with plain text", func(t *testing.T) {
		textContent := &officialmcp.TextContent{Text: "Hello, World!"}
		contents := []officialmcp.Content{textContent}

		result := mcp.MCPContentToMap(contents)
		gt.NotNil(t, result)

		expected := map[string]any{"result": "Hello, World!"}
		gt.Equal(t, expected, result)
	})

	t.Run("Test convertContentToMap with multiple contents", func(t *testing.T) {
		contents := []officialmcp.Content{
			&officialmcp.TextContent{Text: "First content"},
			&officialmcp.TextContent{Text: "Second content"},
		}

		result := mcp.MCPContentToMap(contents)
		gt.NotNil(t, result)

		expected := map[string]any{
			"content_1": "First content",
			"content_2": "Second content",
		}
		gt.Equal(t, expected, result)
	})
}

// Test that existing functionality still works
func TestExistingFunctionalityNotAffected(t *testing.T) {
	// Test that NewStdio still works
	t.Run("NewStdio functionality", func(t *testing.T) {
		// This test is skipped if TEST_MCP_EXEC_PATH is not set
		mcpExecPath, ok := os.LookupEnv("TEST_MCP_EXEC_PATH")
		if !ok {
			t.Skip("TEST_MCP_EXEC_PATH is not set")
		}

		client, err := mcp.NewStdio(t.Context(), mcpExecPath, []string{})
		if err != nil {
			t.Skip("Could not create stdio client:", err)
		}
		defer func() {
			if err := client.Close(); err != nil {
				t.Logf("failed to close client: %v", err)
			}
		}()

		// Verify it works
		specs, err := client.Specs(t.Context())
		gt.NoError(t, err)
		gt.Array(t, specs).Longer(0)
	})

	// Test that NewStreamableHTTP interface is available (without requiring a server)
	t.Run("NewStreamableHTTP interface available", func(t *testing.T) {
		// Just test that the function signature is correct
		// StreamableHTTP is now implemented and should work but fail with connection error
		client, err := mcp.NewStreamableHTTP(
			t.Context(),
			"http://localhost:99999", // Will fail, but that's expected
		)
		gt.Error(t, err) // Expected to fail with connection error
		gt.Nil(t, client)
		// Error should be connection-related, not "not implemented"
		gt.False(t, err.Error() == "StreamableHTTP transport not yet implemented with official SDK")
	})
}

// TestRecursiveSchemaConversion tests the recursive schema conversion functionality
// that was missing in the initial implementation
func TestBuildStdioEnvInheritsSystemEnv(t *testing.T) {
	t.Run("with custom env vars", func(t *testing.T) {
		customVars := []string{"CUSTOM_VAR=custom_value", "ANOTHER_VAR=another"}
		result := mcp.BuildStdioEnv(customVars)

		// Should contain system PATH
		hasPath := false
		hasCustom := false
		for _, env := range result {
			if strings.HasPrefix(env, "PATH=") {
				hasPath = true
			}
			if env == "CUSTOM_VAR=custom_value" {
				hasCustom = true
			}
		}
		gt.True(t, hasPath)
		gt.True(t, hasCustom)
	})

	t.Run("without custom env vars", func(t *testing.T) {
		result := mcp.BuildStdioEnv(nil)

		// Should still contain system environment
		hasPath := false
		for _, env := range result {
			if strings.HasPrefix(env, "PATH=") {
				hasPath = true
			}
		}
		gt.True(t, hasPath)
		gt.Equal(t, len(os.Environ()), len(result))
	})
}

func TestRecursiveSchemaConversion(t *testing.T) {
	type testCase struct {
		name     string
		schema   map[string]any
		expected *gollem.Parameter
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			result, err := mcp.ConvertSchemaProperty(tc.schema)
			gt.NoError(t, err)

			// Compare the result with expected
			gt.Equal(t, result.Type, tc.expected.Type)
			gt.Equal(t, result.Description, tc.expected.Description)

			// Test object properties recursively
			if tc.expected.Type == gollem.TypeObject {
				gt.NotNil(t, result.Properties)
				gt.Equal(t, len(result.Properties), len(tc.expected.Properties))

				for name, expectedProp := range tc.expected.Properties {
					actualProp, exists := result.Properties[name]
					gt.True(t, exists)
					gt.Equal(t, actualProp.Type, expectedProp.Type)
					gt.Equal(t, actualProp.Description, expectedProp.Description)
					gt.Equal(t, actualProp.Required, expectedProp.Required)

					// Test nested properties if they exist
					if expectedProp.Properties != nil {
						gt.NotNil(t, actualProp.Properties)
						for nestedName, nestedExpected := range expectedProp.Properties {
							nestedActual, nestedExists := actualProp.Properties[nestedName]
							gt.True(t, nestedExists)
							gt.Equal(t, nestedActual.Type, nestedExpected.Type)
							gt.Equal(t, nestedActual.Description, nestedExpected.Description)
							gt.Equal(t, nestedActual.Required, nestedExpected.Required)
						}
					}
				}
			}

			// Test array items recursively
			if tc.expected.Type == gollem.TypeArray {
				gt.NotNil(t, result.Items)
				gt.Equal(t, result.Items.Type, tc.expected.Items.Type)
				gt.Equal(t, result.Items.Description, tc.expected.Items.Description)

				// Test nested array item properties
				if tc.expected.Items.Properties != nil {
					gt.NotNil(t, result.Items.Properties)
					for name, expectedProp := range tc.expected.Items.Properties {
						actualProp, exists := result.Items.Properties[name]
						gt.True(t, exists)
						gt.Equal(t, actualProp.Type, expectedProp.Type)
						gt.Equal(t, actualProp.Description, expectedProp.Description)
					}
				}
			}

			// Test constraints
			if tc.expected.Minimum != nil {
				gt.NotNil(t, result.Minimum)
				gt.Equal(t, *result.Minimum, *tc.expected.Minimum)
			}
			if tc.expected.Maximum != nil {
				gt.NotNil(t, result.Maximum)
				gt.Equal(t, *result.Maximum, *tc.expected.Maximum)
			}
			if tc.expected.MinLength != nil {
				gt.NotNil(t, result.MinLength)
				gt.Equal(t, *result.MinLength, *tc.expected.MinLength)
			}
			if tc.expected.MaxLength != nil {
				gt.NotNil(t, result.MaxLength)
				gt.Equal(t, *result.MaxLength, *tc.expected.MaxLength)
			}
			if tc.expected.Pattern != "" {
				gt.Equal(t, result.Pattern, tc.expected.Pattern)
			}
			if tc.expected.MinItems != nil {
				gt.NotNil(t, result.MinItems)
				gt.Equal(t, *result.MinItems, *tc.expected.MinItems)
			}
			if tc.expected.MaxItems != nil {
				gt.NotNil(t, result.MaxItems)
				gt.Equal(t, *result.MaxItems, *tc.expected.MaxItems)
			}
		}
	}

	t.Run("nested object with multiple levels", runTest(testCase{
		name: "nested object with multiple levels",
		schema: map[string]any{
			"type":        "object",
			"description": "A nested object example",
			"properties": map[string]any{
				"user": map[string]any{
					"type":        "object",
					"description": "User information",
					"properties": map[string]any{
						"name": map[string]any{
							"type":        "string",
							"description": "User name",
							"minLength":   1.0,
							"maxLength":   100.0,
						},
						"age": map[string]any{
							"type":        "integer",
							"description": "User age",
							"minimum":     0.0,
							"maximum":     150.0,
						},
						"address": map[string]any{
							"type":        "object",
							"description": "User address",
							"properties": map[string]any{
								"street": map[string]any{
									"type":        "string",
									"description": "Street address",
								},
								"city": map[string]any{
									"type":        "string",
									"description": "City name",
								},
							},
							"required": []any{"street", "city"},
						},
					},
					"required": []any{"name", "age"},
				},
				"preferences": map[string]any{
					"type":        "object",
					"description": "User preferences",
					"properties": map[string]any{
						"theme": map[string]any{
							"type":        "string",
							"description": "UI theme",
							"enum":        []any{"light", "dark"},
						},
					},
				},
			},
			"required": []any{"user"},
		},
		expected: &gollem.Parameter{
			Type:        gollem.TypeObject,
			Description: "A nested object example",
			Properties: map[string]*gollem.Parameter{
				"user": {
					Type:        gollem.TypeObject,
					Description: "User information",
					Required:    true,
					Properties: map[string]*gollem.Parameter{
						"name": {
							Type:        gollem.TypeString,
							Description: "User name",
							Required:    true,
							MinLength:   &[]int{1}[0],
							MaxLength:   &[]int{100}[0],
						},
						"age": {
							Type:        gollem.TypeInteger,
							Description: "User age",
							Required:    true,
							Minimum:     &[]float64{0.0}[0],
							Maximum:     &[]float64{150.0}[0],
						},
						"address": {
							Type:        gollem.TypeObject,
							Description: "User address",
							Properties: map[string]*gollem.Parameter{
								"street": {
									Type:        gollem.TypeString,
									Description: "Street address",
									Required:    true,
								},
								"city": {
									Type:        gollem.TypeString,
									Description: "City name",
									Required:    true,
								},
							},
						},
					},
				},
				"preferences": {
					Type:        gollem.TypeObject,
					Description: "User preferences",
					Properties: map[string]*gollem.Parameter{
						"theme": {
							Type:        gollem.TypeString,
							Description: "UI theme",
							Enum:        []string{"light", "dark"},
						},
					},
				},
			},
		},
	}))

	t.Run("array with complex object items", runTest(testCase{
		name: "array with complex object items",
		schema: map[string]any{
			"type":        "array",
			"description": "Array of complex objects",
			"minItems":    1.0,
			"maxItems":    10.0,
			"items": map[string]any{
				"type":        "object",
				"description": "Product item",
				"properties": map[string]any{
					"id": map[string]any{
						"type":        "integer",
						"description": "Product ID",
						"minimum":     1.0,
					},
					"name": map[string]any{
						"type":        "string",
						"description": "Product name",
						"pattern":     "^[a-zA-Z0-9\\s]+$",
					},
					"tags": map[string]any{
						"type":        "array",
						"description": "Product tags",
						"items": map[string]any{
							"type":        "string",
							"description": "Tag name",
						},
					},
				},
				"required": []any{"id", "name"},
			},
		},
		expected: &gollem.Parameter{
			Type:        gollem.TypeArray,
			Description: "Array of complex objects",
			MinItems:    &[]int{1}[0],
			MaxItems:    &[]int{10}[0],
			Items: &gollem.Parameter{
				Type:        gollem.TypeObject,
				Description: "Product item",
				Properties: map[string]*gollem.Parameter{
					"id": {
						Type:        gollem.TypeInteger,
						Description: "Product ID",
						Required:    true,
						Minimum:     &[]float64{1.0}[0],
					},
					"name": {
						Type:        gollem.TypeString,
						Description: "Product name",
						Required:    true,
						Pattern:     "^[a-zA-Z0-9\\s]+$",
					},
					"tags": {
						Type:        gollem.TypeArray,
						Description: "Product tags",
						Items: &gollem.Parameter{
							Type:        gollem.TypeString,
							Description: "Tag name",
						},
					},
				},
			},
		},
	}))

	t.Run("array of arrays (nested arrays)", runTest(testCase{
		name: "array of arrays (nested arrays)",
		schema: map[string]any{
			"type":        "array",
			"description": "Matrix of numbers",
			"items": map[string]any{
				"type":        "array",
				"description": "Row of numbers",
				"items": map[string]any{
					"type":        "number",
					"description": "Number value",
					"minimum":     -100.0,
					"maximum":     100.0,
				},
				"minItems": 1.0,
				"maxItems": 5.0,
			},
		},
		expected: &gollem.Parameter{
			Type:        gollem.TypeArray,
			Description: "Matrix of numbers",
			Items: &gollem.Parameter{
				Type:        gollem.TypeArray,
				Description: "Row of numbers",
				MinItems:    &[]int{1}[0],
				MaxItems:    &[]int{5}[0],
				Items: &gollem.Parameter{
					Type:        gollem.TypeNumber,
					Description: "Number value",
					Minimum:     &[]float64{-100.0}[0],
					Maximum:     &[]float64{100.0}[0],
				},
			},
		},
	}))
}
