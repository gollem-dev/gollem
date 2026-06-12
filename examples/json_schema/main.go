package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/gollem-dev/gollem/llm/gemini"
	"github.com/gollem-dev/gollem/llm/openai"
)

// UserProfile demonstrates struct-based schema with various constraints
type UserProfile struct {
	Name      string   `json:"name" description:"Full name of the user" required:"true"`
	Age       int      `json:"age" description:"Age in years" min:"0" max:"150"`
	Email     string   `json:"email" description:"Email address" required:"true"`
	Interests []string `json:"interests" description:"List of interests or hobbies"`
	Location  Location `json:"location" description:"User's location"`
}

type Location struct {
	City    string `json:"city" description:"City name"`
	Country string `json:"country" description:"Country name"`
}

// createUserProfileSchema creates a schema for extracting user profile information
// This example demonstrates manual schema construction
func createUserProfileSchema() *gollem.Parameter {
	return &gollem.Parameter{
		Title:       "UserProfile",
		Description: "Structured user profile information",
		Type:        gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"name": {
				Type:        gollem.TypeString,
				Description: "Full name of the user",
				Required:    true,
			},
			"age": {
				Type:        gollem.TypeInteger,
				Description: "Age in years",
				Minimum:     Ptr(0.0),
				Maximum:     Ptr(150.0),
			},
			"email": {
				Type:        gollem.TypeString,
				Description: "Email address",
				Required:    true,
			},
			"interests": {
				Type: gollem.TypeArray,
				Items: &gollem.Parameter{
					Type: gollem.TypeString,
				},
				Description: "List of interests or hobbies",
			},
			"location": {
				Type: gollem.TypeObject,
				Properties: map[string]*gollem.Parameter{
					"city": {
						Type:        gollem.TypeString,
						Description: "City name",
					},
					"country": {
						Type:        gollem.TypeString,
						Description: "Country name",
					},
				},
				Description: "User's location",
			},
		},
	}
}

// createUserProfileSchemaFromStruct creates a schema using ToSchema
// This is the recommended approach for most use cases
func createUserProfileSchemaFromStruct() (*gollem.Parameter, error) {
	schema, err := gollem.ToSchema(UserProfile{})
	if err != nil {
		return nil, err
	}
	schema.Title = "UserProfile"
	schema.Description = "Structured user profile information"
	return schema, nil
}

// Ptr returns a pointer to a value of any type
func Ptr[T any](v T) *T {
	return &v
}

// prettyPrintJSON parses and pretty-prints JSON string
func prettyPrintJSON(jsonStr string) (string, error) {
	var data map[string]any
	if err := json.Unmarshal([]byte(jsonStr), &data); err != nil {
		return "", fmt.Errorf("failed to parse JSON: %w", err)
	}

	pretty, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal JSON: %w", err)
	}

	return string(pretty), nil
}

func runOpenAIExample() error {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("OPENAI_API_KEY is not set")
	}

	ctx := context.Background()
	client, err := openai.New(ctx, apiKey)
	if err != nil {
		return fmt.Errorf("failed to create OpenAI client: %w", err)
	}

	schema := createUserProfileSchema()

	session, err := client.NewSession(ctx,
		gollem.WithSessionContentType(gollem.ContentTypeJSON),
		gollem.WithSessionResponseSchema(schema),
	)
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}

	resp, err := session.Generate(ctx, []gollem.Input{
		gollem.Text("Extract user information: Sarah Johnson is 28 years old, email: sarah.j@example.com, lives in Seattle, USA, and enjoys hiking, photography, and cooking."),
	})
	if err != nil {
		return fmt.Errorf("failed to generate content: %w", err)
	}

	prettyJSON, err := prettyPrintJSON(resp.Texts[0])
	if err != nil {
		return err
	}

	fmt.Println("=== OpenAI Result ===")
	fmt.Println(prettyJSON)
	fmt.Println()

	return nil
}

func runClaudeExample() error {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("ANTHROPIC_API_KEY is not set")
	}

	ctx := context.Background()
	client, err := claude.New(ctx, apiKey)
	if err != nil {
		return fmt.Errorf("failed to create Claude client: %w", err)
	}

	schema := createUserProfileSchema()

	session, err := client.NewSession(ctx,
		gollem.WithSessionContentType(gollem.ContentTypeJSON),
		gollem.WithSessionResponseSchema(schema),
	)
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}

	resp, err := session.Generate(ctx, []gollem.Input{
		gollem.Text("Extract user information: Michael Chen is 35 years old, email: m.chen@tech.com, lives in San Francisco, USA, and enjoys programming, gaming, and traveling."),
	})
	if err != nil {
		return fmt.Errorf("failed to generate content: %w", err)
	}

	prettyJSON, err := prettyPrintJSON(resp.Texts[0])
	if err != nil {
		return err
	}

	fmt.Println("=== Claude Result ===")
	fmt.Println(prettyJSON)
	fmt.Println()

	return nil
}

func runGeminiExample() error {
	projectID := os.Getenv("GEMINI_PROJECT_ID")
	location := os.Getenv("GEMINI_LOCATION")
	if projectID == "" || location == "" {
		return fmt.Errorf("GEMINI_PROJECT_ID or GEMINI_LOCATION is not set")
	}

	ctx := context.Background()
	client, err := gemini.New(ctx, projectID, location)
	if err != nil {
		return fmt.Errorf("failed to create Gemini client: %w", err)
	}

	schema := createUserProfileSchema()

	session, err := client.NewSession(ctx,
		gollem.WithSessionContentType(gollem.ContentTypeJSON),
		gollem.WithSessionResponseSchema(schema),
	)
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}

	resp, err := session.Generate(ctx, []gollem.Input{
		gollem.Text("Extract user information: Emily Davis is 31 years old, email: emily.d@design.com, lives in Portland, USA, and enjoys painting, yoga, and reading."),
	})
	if err != nil {
		return fmt.Errorf("failed to generate content: %w", err)
	}

	prettyJSON, err := prettyPrintJSON(resp.Texts[0])
	if err != nil {
		return err
	}

	fmt.Println("=== Gemini Result ===")
	fmt.Println(prettyJSON)
	fmt.Println()

	return nil
}

func runStructToSchemaExample() error {
	fmt.Println("=== Struct-to-Schema Conversion Example ===")

	// Convert struct to Parameter
	param, err := gollem.ToSchema(UserProfile{})
	if err != nil {
		return fmt.Errorf("failed to convert struct to schema: %w", err)
	}

	paramJSON, err := json.MarshalIndent(param, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal schema to JSON: %w", err)
	}
	fmt.Println("Generated Schema from UserProfile struct:")
	fmt.Println(string(paramJSON))
	fmt.Println()

	// Create Parameter from struct with Title and Description
	schema, err := createUserProfileSchemaFromStruct()
	if err != nil {
		return fmt.Errorf("failed to create schema: %w", err)
	}

	fmt.Printf("Schema Title: %s\n", schema.Title)
	fmt.Printf("Description: %s\n", schema.Description)
	fmt.Printf("Required fields: %v\n", schema.Required)
	fmt.Println()

	return nil
}

func main() {
	fmt.Println("JSON Schema Example")
	fmt.Println("===================")
	fmt.Println()

	// Demonstrate struct-to-schema conversion
	if err := runStructToSchemaExample(); err != nil {
		log.Printf("Struct-to-schema example failed: %v", err)
	}

	// Try each provider
	if err := runOpenAIExample(); err != nil {
		log.Printf("OpenAI example skipped: %v", err)
	}

	if err := runClaudeExample(); err != nil {
		log.Printf("Claude example skipped: %v", err)
	}

	if err := runGeminiExample(); err != nil {
		log.Printf("Gemini example skipped: %v", err)
	}
}
