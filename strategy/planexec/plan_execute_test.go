package planexec_test

import (
	"context"
	"os"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/gollem-dev/gollem/llm/gemini"
	"github.com/gollem-dev/gollem/llm/openai"
	"github.com/gollem-dev/gollem/mock"
	"github.com/gollem-dev/gollem/strategy/planexec"
	"github.com/m-mizutani/goerr/v2"
	"github.com/m-mizutani/gt"
)

// testHooks is a test implementation of PlanExecuteHooks
type testHooks struct {
	onPlanCreated func(ctx context.Context, plan *planexec.Plan) error
	onPlanUpdated func(ctx context.Context, plan *planexec.Plan) error
	onTaskDone    func(ctx context.Context, plan *planexec.Plan, task *planexec.Task) error
}

func (h *testHooks) OnPlanCreated(ctx context.Context, plan *planexec.Plan) error {
	if h.onPlanCreated != nil {
		return h.onPlanCreated(ctx, plan)
	}
	return nil
}

func (h *testHooks) OnPlanUpdated(ctx context.Context, plan *planexec.Plan) error {
	if h.onPlanUpdated != nil {
		return h.onPlanUpdated(ctx, plan)
	}
	return nil
}

func (h *testHooks) OnTaskDone(ctx context.Context, plan *planexec.Plan, task *planexec.Task) error {
	if h.onTaskDone != nil {
		return h.onTaskDone(ctx, plan, task)
	}
	return nil
}

// testTool is a simple implementation of gollem.Tool for testing
type testTool struct {
	name        string
	description string
	parameters  map[string]*gollem.Parameter
	runFunc     func(ctx context.Context, args map[string]any) (map[string]any, error)
}

func (t *testTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        t.name,
		Description: t.description,
		Parameters:  t.parameters,
	}
}

func (t *testTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return t.runFunc(ctx, args)
}

func TestBasicPlanExecution(t *testing.T) {
	// Helper function to create mock client for direct response
	createDirectResponseMock := func(response string) *mock.LLMClientMock {
		return &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						jsonResponse := `{
							"needs_plan": false,
							"direct_response": "` + response + `"
						}`
						return &gollem.Response{
							Texts: []string{jsonResponse},
						}, nil
					},
					HistoryFunc: func() (*gollem.History, error) {
						return &gollem.History{}, nil
					},
				}, nil
			},
		}
	}

	// Helper function to create mock client for plan execution
	createPlanExecutionMock := func() *mock.LLMClientMock {
		callCount := 0
		return &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++
						switch callCount {
						case 1:
							// First call: return a plan
							return &gollem.Response{
								Texts: []string{`{
									"needs_plan": true,
									"user_intent": "Want to know the result of 10 + 5",
									"goal": "Calculate 10 + 5",
									"tasks": [{"description": "Add 10 and 5", "state": "pending"}]
								}`},
							}, nil
						case 2:
							// Second call: task execution
							return &gollem.Response{
								Texts: []string{"The result is 15"},
							}, nil
						case 3:
							// Third call: reflection to complete
							return &gollem.Response{
								Texts: []string{`{
									"new_tasks": [],
									"updated_tasks": [],
									"reason": "All tasks completed."
								}`},
							}, nil
						default:
							// Fourth call: final conclusion
							return &gollem.Response{
								Texts: []string{"The calculation is complete. The result is 15."},
							}, nil
						}
					},
					HistoryFunc: func() (*gollem.History, error) {
						return &gollem.History{}, nil
					},
				}, nil
			},
		}
	}

	t.Run("Plan with single task execution", func(t *testing.T) {
		ctx := context.Background()
		mockClient := createPlanExecutionMock()

		// Track task execution
		var planCreatedCalled int32
		var taskDoneCalled int32
		var createdPlan *planexec.Plan
		var completedTask *planexec.Task

		hooks := &testHooks{
			onPlanCreated: func(ctx context.Context, plan *planexec.Plan) error {
				atomic.AddInt32(&planCreatedCalled, 1)
				createdPlan = plan
				return nil
			},
			onTaskDone: func(ctx context.Context, plan *planexec.Plan, task *planexec.Task) error {
				atomic.AddInt32(&taskDoneCalled, 1)
				completedTask = task
				return nil
			},
		}

		// Create strategy with hooks
		strategy := planexec.New(mockClient, planexec.WithHooks(hooks))

		// Create agent and test
		agent := gollem.New(mockClient, gollem.WithStrategy(strategy))
		resp, err := agent.Execute(ctx, gollem.Text("Calculate 10 + 5"))
		gt.NoError(t, err)
		gt.V(t, resp).NotNil()
		gt.True(t, len(resp.Texts) > 0)

		// Verify plan was created with tasks
		gt.V(t, atomic.LoadInt32(&planCreatedCalled)).Equal(int32(1))
		gt.V(t, createdPlan).NotNil()
		gt.V(t, len(createdPlan.Tasks)).Equal(1)
		gt.S(t, createdPlan.Tasks[0].Description).Contains("Add 10 and 5")

		// Verify OnTaskDone was called
		gt.V(t, atomic.LoadInt32(&taskDoneCalled)).Equal(int32(1))
		gt.V(t, completedTask).NotNil()
		gt.V(t, completedTask.State).Equal(planexec.TaskStateCompleted)
	})

	t.Run("Direct response without tasks", func(t *testing.T) {
		ctx := context.Background()
		mockClient := createDirectResponseMock("The answer is 4")

		// Track plan creation
		var planCreatedCalled int32
		var createdPlan *planexec.Plan

		hooks := &testHooks{
			onPlanCreated: func(ctx context.Context, plan *planexec.Plan) error {
				atomic.AddInt32(&planCreatedCalled, 1)
				createdPlan = plan
				return nil
			},
		}

		// Create strategy with hooks
		strategy := planexec.New(mockClient, planexec.WithHooks(hooks))

		// Create agent and test
		agent := gollem.New(mockClient, gollem.WithStrategy(strategy))
		resp, err := agent.Execute(ctx, gollem.Text("What is 2 + 2?"))
		gt.NoError(t, err)
		gt.V(t, resp).NotNil()
		gt.V(t, resp.Texts[0]).Equal("The answer is 4")

		// Verify plan was created but with no tasks (direct response)
		gt.V(t, atomic.LoadInt32(&planCreatedCalled)).Equal(int32(1))
		gt.V(t, createdPlan).NotNil()
		gt.V(t, len(createdPlan.Tasks)).Equal(0)
		gt.V(t, createdPlan.DirectResponse).Equal("The answer is 4")
	})

	t.Run("Comprehensive test with Hooks and Middleware", func(t *testing.T) {
		ctx := context.Background()

		// Track hook calls
		var planCreatedCalled int32
		var planUpdatedCalled int32
		var taskDoneCalled int32
		var createdPlan *planexec.Plan
		var updatedPlan *planexec.Plan

		// Track middleware calls
		var middlewareApplied int32

		// Create hooks
		hooks := &testHooks{
			onPlanCreated: func(ctx context.Context, plan *planexec.Plan) error {
				atomic.AddInt32(&planCreatedCalled, 1)
				createdPlan = plan
				return nil
			},
			onPlanUpdated: func(ctx context.Context, plan *planexec.Plan) error {
				atomic.AddInt32(&planUpdatedCalled, 1)
				updatedPlan = plan
				return nil
			},
			onTaskDone: func(ctx context.Context, plan *planexec.Plan, task *planexec.Task) error {
				atomic.AddInt32(&taskDoneCalled, 1)
				return nil
			},
		}

		// Create middleware
		mockMiddleware := func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
			return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
				return next(ctx, req)
			}
		}

		// Create mock client that tests plan updates
		callCount := 0
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				// Apply options to check middleware is passed
				cfg := &gollem.SessionConfig{}
				for _, opt := range options {
					opt(cfg)
				}

				// Check if middleware was applied to the session
				if len(cfg.ContentBlockMiddlewares()) > 0 {
					atomic.AddInt32(&middlewareApplied, 1)
				}

				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++
						switch callCount {
						case 1:
							// First call: return a plan with 2 tasks
							return &gollem.Response{
								Texts: []string{`{
									"needs_plan": true,
									"user_intent": "Want to know the data processing results",
									"goal": "Process data",
									"tasks": [
										{"description": "Load data"},
										{"description": "Transform data"}
									]
								}`},
							}, nil
						case 2:
							// Execute first task
							return &gollem.Response{
								Texts: []string{"Data loaded successfully"},
							}, nil
						case 3:
							// Reflection with plan update - add new task
							return &gollem.Response{
								Texts: []string{`{
									"new_tasks": ["Validate data"],
									"updated_tasks": [],
									"reason": "Need additional validation step"
								}`},
							}, nil
						case 4:
							// Execute second task
							return &gollem.Response{
								Texts: []string{"Data transformed"},
							}, nil
						case 5:
							// Reflection - no updates needed
							return &gollem.Response{
								Texts: []string{`{
									"new_tasks": [],
									"updated_tasks": [],
									"reason": "All tasks on track"
								}`},
							}, nil
						case 6:
							// Execute validation task
							return &gollem.Response{
								Texts: []string{"Data validated"},
							}, nil
						case 7:
							// Final reflection - no more tasks
							return &gollem.Response{
								Texts: []string{`{
									"new_tasks": [],
									"updated_tasks": [],
									"reason": "All tasks completed"
								}`},
							}, nil
						default:
							// Final conclusion
							return &gollem.Response{
								Texts: []string{"All data processing tasks completed successfully"},
							}, nil
						}
					},
					HistoryFunc: func() (*gollem.History, error) {
						return &gollem.History{}, nil
					},
				}, nil
			},
		}

		// Create strategy with all options
		strategy := planexec.New(mockClient,
			planexec.WithHooks(hooks),
			planexec.WithMiddleware(mockMiddleware),
		)

		// Create agent and test
		agent := gollem.New(mockClient, gollem.WithStrategy(strategy))
		resp, err := agent.Execute(ctx, gollem.Text("Process data"))
		gt.NoError(t, err)
		gt.V(t, resp).NotNil()

		// Verify all hooks were called
		gt.V(t, atomic.LoadInt32(&planCreatedCalled)).Equal(int32(1))
		gt.V(t, atomic.LoadInt32(&planUpdatedCalled)).Equal(int32(1)) // Should be called when new task is added
		gt.V(t, createdPlan).NotNil()
		gt.V(t, createdPlan.Goal).Equal("Process data")
		gt.V(t, updatedPlan).NotNil()
		gt.V(t, len(updatedPlan.Tasks)).Equal(3) // Original 2 tasks + 1 new task

		// Verify OnTaskDone was called for each completed task (3 tasks total)
		gt.V(t, atomic.LoadInt32(&taskDoneCalled)).Equal(int32(3))

		// Verify middleware was applied
		gt.True(t, atomic.LoadInt32(&middlewareApplied) > 0) // Should be applied at least once
	})
}

// Test with real LLM providers using Agent.Execute
func TestPlanExecuteWithLLMs(t *testing.T) {
	// Create test tools for the agent to use
	createTestTools := func() []gollem.Tool {
		// Tool to get weather information
		getWeather := &testTool{
			name:        "get_weather",
			description: "Get current weather for a city",
			parameters: map[string]*gollem.Parameter{
				"city": {
					Type:        gollem.TypeString,
					Description: "City name",
					Required:    true,
				},
			},
			runFunc: func(ctx context.Context, args map[string]any) (map[string]any, error) {
				city, ok := args["city"].(string)
				if !ok || city == "" {
					return nil, goerr.New("missing or invalid 'city' parameter")
				}
				return map[string]any{
					"city":        city,
					"temperature": 22,
					"condition":   "sunny",
				}, nil
			},
		}

		// Tool to calculate distance
		calculateDistance := &testTool{
			name:        "calculate_distance",
			description: "Calculate distance between two cities in km",
			parameters: map[string]*gollem.Parameter{
				"from": {
					Type:        gollem.TypeString,
					Description: "Starting city",
					Required:    true,
				},
				"to": {
					Type:        gollem.TypeString,
					Description: "Destination city",
					Required:    true,
				},
			},
			runFunc: func(ctx context.Context, args map[string]any) (map[string]any, error) {
				from, ok := args["from"].(string)
				if !ok || from == "" {
					return nil, goerr.New("missing or invalid 'from' parameter")
				}
				to, ok := args["to"].(string)
				if !ok || to == "" {
					return nil, goerr.New("missing or invalid 'to' parameter")
				}
				// Mock distance calculation
				distance := len(from) + len(to)*10 // Simple mock calculation
				return map[string]any{
					"from":     from,
					"to":       to,
					"distance": distance,
				}, nil
			},
		}

		// Tool to search database
		searchDB := &testTool{
			name:        "search_database",
			description: "Search information in database",
			parameters: map[string]*gollem.Parameter{
				"query": {
					Type:        gollem.TypeString,
					Description: "Search query",
					Required:    true,
				},
			},
			runFunc: func(ctx context.Context, args map[string]any) (map[string]any, error) {
				query, ok := args["query"].(string)
				if !ok || query == "" {
					return nil, goerr.New("missing or invalid 'query' parameter")
				}
				return map[string]any{
					"query":   query,
					"results": []string{"result1", "result2", "result3"},
					"count":   3,
				}, nil
			},
		}

		return []gollem.Tool{getWeather, calculateDistance, searchDB}
	}

	// Helper function for testing with Agent.Execute
	testWithAgent := func(client gollem.LLMClient, _ string) func(t *testing.T) {
		return func(t *testing.T) {
			ctx := context.Background()

			// Test with multiple tool calls
			var toolCallCount int32
			var toolNames []string

			// Track plan creation and updates
			var planCreatedCalled int32
			var createdPlan *planexec.Plan
			var taskCount int

			hooks := &testHooks{
				onPlanCreated: func(ctx context.Context, plan *planexec.Plan) error {
					atomic.AddInt32(&planCreatedCalled, 1)
					createdPlan = plan
					taskCount = len(plan.Tasks)
					return nil
				},
			}

			// Create middleware to track tool calls
			toolTracker := func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
				return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
					resp, err := next(ctx, req)
					if err != nil {
						return resp, err
					}

					// Track tool calls in response
					if resp != nil && resp.FunctionCalls != nil {
						for _, call := range resp.FunctionCalls {
							atomic.AddInt32(&toolCallCount, 1)
							toolNames = append(toolNames, call.Name)
						}
					}

					return resp, nil
				}
			}

			// Create strategy with hooks and middleware
			strategy := planexec.New(client,
				planexec.WithHooks(hooks),
				planexec.WithMiddleware(toolTracker),
			)

			// Create test tools
			tools := createTestTools()

			// Create agent with the strategy and tools
			agent := gollem.New(client,
				gollem.WithStrategy(strategy),
				gollem.WithTools(tools...),
				gollem.WithContentBlockMiddleware(toolTracker),
			)

			// Execute task that requires multiple tool calls
			response, err := agent.Execute(ctx,
				gollem.Text("Get the weather for Tokyo and calculate the distance from Tokyo to Osaka"))
			gt.NoError(t, err)
			gt.V(t, response).NotNil()
			if response == nil {
				return // Skip if API failed
			}
			gt.True(t, len(response.Texts) > 0)

			// Verify plan was created
			gt.V(t, atomic.LoadInt32(&planCreatedCalled)).Equal(int32(1))
			gt.V(t, createdPlan).NotNil()

			// Verify tasks were created (should have at least 1 task for multi-step operation)
			gt.True(t, taskCount > 0)

			// Verify tools were called (should be called at least 2 times for the 2 tools)
			finalToolCallCount := atomic.LoadInt32(&toolCallCount)
			gt.True(t, finalToolCallCount >= 2)

			// Verify expected tools were called
			toolNameStr := strings.Join(toolNames, ",")
			gt.S(t, toolNameStr).Contains("get_weather")
			gt.S(t, toolNameStr).Contains("calculate_distance")

			// Verify response contains results from tools
			responseText := strings.ToLower(strings.Join(response.Texts, " "))
			gt.S(t, responseText).ContainsAny("tokyo", "weather", "distance", "osaka")
		}
	}

	// Test with OpenAI
	t.Run("OpenAI", func(t *testing.T) {
		t.Parallel()
		apiKey := os.Getenv("TEST_OPENAI_API_KEY")
		if apiKey == "" {
			t.Skip("TEST_OPENAI_API_KEY or OPENAI_API_KEY is not set")
		}

		client, err := openai.New(context.Background(), apiKey)
		gt.NoError(t, err)

		testWithAgent(client, "OpenAI")(t)
	})

	// Test with Claude
	t.Run("Claude", func(t *testing.T) {
		t.Parallel()
		apiKey := os.Getenv("TEST_CLAUDE_API_KEY")
		if apiKey == "" {
			t.Skip("TEST_CLAUDE_API_KEY or ANTHROPIC_API_KEY is not set")
		}

		client, err := claude.New(context.Background(), apiKey)
		gt.NoError(t, err)

		testWithAgent(client, "Claude")(t)
	})

	// Test with Gemini
	t.Run("Gemini", func(t *testing.T) {
		t.Parallel()
		projectID := os.Getenv("TEST_GCP_PROJECT_ID")
		location := os.Getenv("TEST_GCP_LOCATION")

		if projectID == "" || location == "" {
			t.Skip("Required Gemini env vars not set")
		}

		client, err := gemini.New(context.Background(), projectID, location)
		gt.NoError(t, err)

		testWithAgent(client, "Gemini")(t)
	})
}

func TestExternalPlanGeneration(t *testing.T) {
	ctx := context.Background()

	t.Run("GeneratePlan with basic parameters", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						return &gollem.Response{
							Texts: []string{`{
								"needs_plan": true,
								"user_intent": "Calculate sum",
								"goal": "Add two numbers",
								"tasks": [{"description": "Perform addition"}]
							}`},
						}, nil
					},
					HistoryFunc: func() (*gollem.History, error) {
						return &gollem.History{}, nil
					},
				}, nil
			},
		}

		plan, err := planexec.GeneratePlan(ctx, mockClient, []gollem.Input{gollem.Text("Calculate 10 + 5")}, nil, "", nil)
		gt.NoError(t, err)
		gt.V(t, plan).NotNil()
		gt.V(t, plan.Goal).Equal("Add two numbers")
		gt.V(t, len(plan.Tasks)).Equal(1)
		gt.V(t, plan.Tasks[0].Description).Equal("Perform addition")
	})

	t.Run("GeneratePlan with tools and system prompt", func(t *testing.T) {
		tool := &testTool{
			name:        "test_tool",
			description: "A test tool",
			runFunc:     func(ctx context.Context, args map[string]any) (map[string]any, error) { return nil, nil },
		}

		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						return &gollem.Response{
							Texts: []string{`{
								"needs_plan": true,
								"user_intent": "Test intent",
								"goal": "Test goal",
								"tasks": [{"description": "Test task"}]
							}`},
						}, nil
					},
					HistoryFunc: func() (*gollem.History, error) {
						return &gollem.History{}, nil
					},
				}, nil
			},
		}

		plan, err := planexec.GeneratePlan(
			ctx,
			mockClient,
			[]gollem.Input{gollem.Text("Test")},
			[]gollem.Tool{tool},
			"You are a test assistant",
			nil,
		)
		gt.NoError(t, err)
		gt.V(t, plan).NotNil()
		gt.V(t, plan.Goal).Equal("Test goal")
	})

	t.Run("GeneratePlan error cases", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}

		// Nil client
		_, err := planexec.GeneratePlan(ctx, nil, []gollem.Input{gollem.Text("Test")}, nil, "", nil)
		gt.Error(t, err)
		gt.S(t, err.Error()).Contains("client is required")

		// Empty inputs
		_, err = planexec.GeneratePlan(ctx, mockClient, []gollem.Input{}, nil, "", nil)
		gt.Error(t, err)
		gt.S(t, err.Error()).Contains("inputs are required")
	})
}

func TestWithPlanOption(t *testing.T) {
	ctx := context.Background()

	t.Run("Strategy with pre-generated plan", func(t *testing.T) {
		// Create a pre-generated plan
		prePlan := &planexec.Plan{
			UserQuestion: "What is 2 + 2?",
			UserIntent:   "Get calculation result",
			Goal:         "Calculate 2 + 2",
			Tasks: []planexec.Task{
				{
					ID:          "task-1",
					Description: "Add 2 and 2",
					State:       planexec.TaskStatePending,
				},
			},
		}

		// Track hook calls
		var planCreatedCalled int32
		var taskDoneCalled int32
		hooks := &testHooks{
			onPlanCreated: func(ctx context.Context, plan *planexec.Plan) error {
				atomic.AddInt32(&planCreatedCalled, 1)
				// Verify it's our pre-generated plan
				gt.V(t, plan.Goal).Equal("Calculate 2 + 2")
				return nil
			},
			onTaskDone: func(ctx context.Context, plan *planexec.Plan, task *planexec.Task) error {
				atomic.AddInt32(&taskDoneCalled, 1)
				return nil
			},
		}

		// Create mock client (should not be called for planning)
		callCount := 0
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++
						switch callCount {
						case 1:
							// First call: task execution (planning should be skipped)
							return &gollem.Response{
								Texts: []string{"The result is 4"},
							}, nil
						case 2:
							// Second call: reflection
							return &gollem.Response{
								Texts: []string{`{
									"new_tasks": [],
									"updated_tasks": [],
									"reason": "Task completed"
								}`},
							}, nil
						default:
							// Third call: conclusion
							return &gollem.Response{
								Texts: []string{"Calculation complete. The answer is 4."},
							}, nil
						}
					},
					HistoryFunc: func() (*gollem.History, error) {
						return &gollem.History{}, nil
					},
				}, nil
			},
		}

		// Create strategy with pre-generated plan
		strategy := planexec.New(mockClient,
			planexec.WithPlan(prePlan),
			planexec.WithHooks(hooks),
		)

		// Create agent and execute
		agent := gollem.New(mockClient, gollem.WithStrategy(strategy))
		resp, err := agent.Execute(ctx, gollem.Text("Calculate 2 + 2"))
		gt.NoError(t, err)
		gt.V(t, resp).NotNil()

		// Verify OnPlanCreated was called exactly once
		gt.V(t, atomic.LoadInt32(&planCreatedCalled)).Equal(int32(1))

		// Verify OnTaskDone was called
		gt.V(t, atomic.LoadInt32(&taskDoneCalled)).Equal(int32(1))

		// Verify planning session was NOT created (only 3 calls: execute, reflect, conclude)
		gt.V(t, callCount).Equal(3)
	})

	t.Run("Strategy with pre-generated plan - direct response", func(t *testing.T) {
		// Create a plan with no tasks (direct response)
		prePlan := &planexec.Plan{
			DirectResponse: "The answer is 42",
			Tasks:          []planexec.Task{},
		}

		var planCreatedCalled int32
		hooks := &testHooks{
			onPlanCreated: func(ctx context.Context, plan *planexec.Plan) error {
				atomic.AddInt32(&planCreatedCalled, 1)
				return nil
			},
		}

		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						t.Fatal("LLM should not be called when using direct response plan")
						return nil, nil
					},
					HistoryFunc: func() (*gollem.History, error) {
						return &gollem.History{}, nil
					},
				}, nil
			},
		}

		strategy := planexec.New(mockClient,
			planexec.WithPlan(prePlan),
			planexec.WithHooks(hooks),
		)

		agent := gollem.New(mockClient, gollem.WithStrategy(strategy))
		resp, err := agent.Execute(ctx, gollem.Text("Test"))
		gt.NoError(t, err)
		gt.V(t, resp).NotNil()
		gt.V(t, resp.Texts[0]).Equal("The answer is 42")

		// Verify OnPlanCreated was called
		gt.V(t, atomic.LoadInt32(&planCreatedCalled)).Equal(int32(1))
	})
}

func TestUserQuestionExtraction(t *testing.T) {
	ctx := context.Background()

	runTest := func(tc struct {
		name             string
		inputs           []gollem.Input
		expectedQuestion string
	}) func(t *testing.T) {
		return func(t *testing.T) {
			// Create mock client
			mockClient := &mock.LLMClientMock{
				NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
					return &mock.SessionMock{
						GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
							// Return direct response (no tasks needed)
							return &gollem.Response{
								Texts: []string{`{
									"needs_plan": false,
									"direct_response": "Test response"
								}`},
							}, nil
						},
						HistoryFunc: func() (*gollem.History, error) {
							return &gollem.History{}, nil
						},
					}, nil
				},
			}

			// Track the created plan
			var createdPlan *planexec.Plan
			hooks := &testHooks{
				onPlanCreated: func(ctx context.Context, plan *planexec.Plan) error {
					createdPlan = plan
					return nil
				},
			}

			// Create strategy with hooks
			strategy := planexec.New(mockClient, planexec.WithHooks(hooks))

			// Create agent and execute
			agent := gollem.New(mockClient, gollem.WithStrategy(strategy))
			_, err := agent.Execute(ctx, tc.inputs...)
			gt.NoError(t, err)

			// Verify user question was extracted
			gt.V(t, createdPlan).NotNil()
			gt.V(t, createdPlan.UserQuestion).Equal(tc.expectedQuestion)
		}
	}

	t.Run("single text input", runTest(struct {
		name             string
		inputs           []gollem.Input
		expectedQuestion string
	}{
		name:             "single text input",
		inputs:           []gollem.Input{gollem.Text("What is the weather in Tokyo?")},
		expectedQuestion: "What is the weather in Tokyo?",
	}))

	t.Run("multiple text inputs", runTest(struct {
		name             string
		inputs           []gollem.Input
		expectedQuestion string
	}{
		name: "multiple text inputs",
		inputs: []gollem.Input{
			gollem.Text("First question"),
			gollem.Text("Second question"),
		},
		expectedQuestion: "First question Second question", // Should combine all text inputs
	}))

	t.Run("empty inputs", runTest(struct {
		name             string
		inputs           []gollem.Input
		expectedQuestion string
	}{
		name:             "empty inputs",
		inputs:           []gollem.Input{},
		expectedQuestion: "", // Should be empty
	}))
}

func TestEnhancedConclusion(t *testing.T) {
	ctx := context.Background()

	runTest := func(tc struct {
		name                 string
		userQuestion         string
		expectedPromptPhrase string
	}) func(t *testing.T) {
		return func(t *testing.T) {
			var capturedPrompt string

			// Create mock client that captures the final conclusion prompt
			callCount := 0
			mockClient := &mock.LLMClientMock{
				NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
					return &mock.SessionMock{
						GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
							callCount++
							switch callCount {
							case 1:
								// First call: return a plan with one task
								return &gollem.Response{
									Texts: []string{`{
										"needs_plan": true,
										"user_intent": "Want to know the test results",
										"goal": "Test goal",
										"tasks": [{"description": "Test task"}]
									}`},
								}, nil
							case 2:
								// Second call: task execution
								return &gollem.Response{
									Texts: []string{"Task completed"},
								}, nil
							case 3:
								// Third call: reflection
								return &gollem.Response{
									Texts: []string{`{
										"new_tasks": [],
										"updated_tasks": [],
										"reason": "All done"
									}`},
								}, nil
							default:
								// Fourth call: final conclusion - capture the prompt
								for _, inp := range input {
									if text, ok := inp.(gollem.Text); ok {
										capturedPrompt = string(text)
									}
								}
								return &gollem.Response{
									Texts: []string{"Final conclusion"},
								}, nil
							}
						},
						HistoryFunc: func() (*gollem.History, error) {
							return &gollem.History{}, nil
						},
					}, nil
				},
			}

			// The user question will be automatically extracted by analyzeAndPlan()
			// when agent.Execute() is called
			strategy := planexec.New(mockClient)

			// Create agent and execute
			agent := gollem.New(mockClient, gollem.WithStrategy(strategy))
			_, err := agent.Execute(ctx, gollem.Text(tc.userQuestion))
			gt.NoError(t, err)

			// Verify the conclusion prompt contains expected phrase
			gt.S(t, capturedPrompt).Contains(tc.expectedPromptPhrase)
		}
	}

	t.Run("with user question", runTest(struct {
		name                 string
		userQuestion         string
		expectedPromptPhrase string
	}{
		name:                 "with user question",
		userQuestion:         "Are there any malicious packages?",
		expectedPromptPhrase: "## User's Original Question", // Should include user question section
	}))

	t.Run("without user question", runTest(struct {
		name                 string
		userQuestion         string
		expectedPromptPhrase string
	}{
		name:                 "without user question",
		userQuestion:         "",
		expectedPromptPhrase: "FINDINGS and RESULTS", // Fallback prompt should still emphasize findings
	}))

	t.Run("prompt includes clear response instruction", runTest(struct {
		name                 string
		userQuestion         string
		expectedPromptPhrase string
	}{
		name:                 "prompt includes clear response instruction",
		userQuestion:         "Did you find the file?",
		expectedPromptPhrase: "Address what the user wants", // Should instruct to address user intent clearly
	}))
}

func TestPlanExec_TaskResultPreservation(t *testing.T) {
	ctx := context.Background()

	t.Run("tool result is preserved in Task.Result", func(t *testing.T) {
		var taskDoneResult string
		var taskDoneCalled int32

		// Create test tool that returns structured data
		queryTool := &testTool{
			name:        "query_database",
			description: "Query database for records",
			parameters: map[string]*gollem.Parameter{
				"query": {
					Type:        gollem.TypeString,
					Description: "Query string",
					Required:    true,
				},
			},
			runFunc: func(ctx context.Context, args map[string]any) (map[string]any, error) {
				return map[string]any{
					"records": []map[string]any{
						{"id": "1", "name": "Alice"},
						{"id": "2", "name": "Bob"},
					},
					"count": 2,
				}, nil
			},
		}

		// Create mock client
		callCount := 0
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++
						switch callCount {
						case 1:
							// Planning phase: return plan with one task
							return &gollem.Response{
								Texts: []string{`{
									"needs_plan": true,
									"user_intent": "Get database records",
									"goal": "Query database",
									"tasks": [{"description": "Query the database"}]
								}`},
							}, nil
						case 2:
							// Task execution: call the tool
							return &gollem.Response{
								FunctionCalls: []*gollem.FunctionCall{
									{
										ID:        "call_1",
										Name:      "query_database",
										Arguments: map[string]any{"query": "SELECT * FROM users"},
									},
								},
							}, nil
						case 3:
							// After tool execution: LLM responds
							return &gollem.Response{
								Texts: []string{"Query executed successfully"},
							}, nil
						case 4:
							// Reflection phase: all done
							return &gollem.Response{
								Texts: []string{`{
									"new_tasks": [],
									"updated_tasks": [],
									"reason": "Task completed"
								}`},
							}, nil
						default:
							// Final conclusion
							return &gollem.Response{
								Texts: []string{"Database query completed"},
							}, nil
						}
					},
					HistoryFunc: func() (*gollem.History, error) {
						return &gollem.History{}, nil
					},
				}, nil
			},
		}

		// Create hooks to capture task result
		hooks := &testHooks{
			onTaskDone: func(ctx context.Context, plan *planexec.Plan, task *planexec.Task) error {
				atomic.AddInt32(&taskDoneCalled, 1)
				taskDoneResult = task.Result
				t.Logf("OnTaskDone called, Task.Result = %s", task.Result)
				return nil
			},
		}

		// Create strategy
		strategy := planexec.New(mockClient, planexec.WithHooks(hooks))

		// Create agent with tool
		agent := gollem.New(mockClient,
			gollem.WithStrategy(strategy),
			gollem.WithTools(queryTool),
		)

		// Execute
		resp, err := agent.Execute(ctx, gollem.Text("Query the database for users"))
		gt.NoError(t, err)
		gt.V(t, resp).NotNil()

		// Verify OnTaskDone was called
		gt.V(t, atomic.LoadInt32(&taskDoneCalled)).Equal(int32(1))

		// CRITICAL TEST: Verify Task.Result contains tool execution result
		gt.V(t, taskDoneResult).NotEqual("")
		gt.S(t, taskDoneResult).Contains("records")
		gt.S(t, taskDoneResult).Contains("Alice")
		gt.S(t, taskDoneResult).Contains("Bob")
		gt.S(t, taskDoneResult).Contains("count")
	})
}

func TestSystemPromptInReflectionAndConclusion(t *testing.T) {
	const systemPrompt = "You are a test assistant with special instructions"

	t.Run("reflection receives system prompt", func(t *testing.T) {
		sessionCallCount := 0
		var reflectionSystemPrompt string

		// Create mock client that captures system prompt during reflection
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				cfg := gollem.NewSessionConfig(options...)
				sessionCallCount++

				// Capture system prompt for reflection session (2nd JSON session)
				if cfg.ContentType() == gollem.ContentTypeJSON && sessionCallCount == 2 {
					reflectionSystemPrompt = cfg.SystemPrompt()
				}

				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						// Planning phase returns plan with one task
						if sessionCallCount == 1 {
							return &gollem.Response{
								Texts: []string{`{
									"needs_plan": true,
									"user_intent": "Test user intent",
									"goal": "Test goal",
									"context_summary": "Test context",
									"constraints": "Test constraints",
									"tasks": [
										{"description": "Task 1"}
									]
								}`},
							}, nil
						}
						// Reflection phase returns no updates
						if sessionCallCount == 2 {
							return &gollem.Response{
								Texts: []string{`{
									"new_tasks": [],
									"updated_tasks": [],
									"reason": "All tasks completed"
								}`},
							}, nil
						}
						// Conclusion phase
						return &gollem.Response{
							Texts: []string{"Final conclusion"},
						}, nil
					},
					HistoryFunc: func() (*gollem.History, error) {
						return &gollem.History{}, nil
					},
				}, nil
			},
		}

		// Create strategy
		strategy := planexec.New(mockClient)
		ctx := context.Background()

		// Initialize strategy
		err := strategy.Init(ctx, []gollem.Input{gollem.Text("Test input")})
		gt.NoError(t, err)

		// Iteration 0: Planning
		state := &gollem.StrategyState{
			InitInput:    []gollem.Input{gollem.Text("Test input")},
			Iteration:    0,
			Tools:        []gollem.Tool{},
			SystemPrompt: systemPrompt,
		}
		inputs, resp, err := strategy.Handle(ctx, state)
		gt.NoError(t, err)
		gt.Nil(t, resp)
		gt.NotNil(t, inputs) // Should return task execution prompt

		// Iteration 1: Execute task (return empty to proceed)
		state.Iteration = 1
		state.NextInput = inputs
		state.LastResponse = nil
		_, _, err = strategy.Handle(ctx, state)
		gt.NoError(t, err)

		// Iteration 2: Task completed, trigger reflection
		state.Iteration = 2
		state.NextInput = nil
		state.LastResponse = &gollem.Response{Texts: []string{"Task 1 result"}}
		_, _, err = strategy.Handle(ctx, state)
		gt.NoError(t, err)

		// Verify reflection was called with system prompt
		gt.Equal(t, systemPrompt, reflectionSystemPrompt)
	})

	t.Run("conclusion receives system prompt", func(t *testing.T) {
		sessionCallCount := 0
		var conclusionSystemPrompt string

		// Create mock client that captures system prompt during conclusion
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				cfg := gollem.NewSessionConfig(options...)
				sessionCallCount++

				// Capture system prompt for conclusion session (non-JSON, should be 3rd call)
				if cfg.ContentType() != gollem.ContentTypeJSON {
					conclusionSystemPrompt = cfg.SystemPrompt()
				}

				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						// Planning phase (1st call)
						if sessionCallCount == 1 {
							return &gollem.Response{
								Texts: []string{`{
									"needs_plan": true,
									"user_intent": "Test intent",
									"goal": "Test goal",
									"context_summary": "Context",
									"constraints": "Constraints",
									"tasks": [
										{"description": "Task 1"}
									]
								}`},
							}, nil
						}
						// Reflection phase (2nd call)
						if sessionCallCount == 2 {
							return &gollem.Response{
								Texts: []string{`{
									"new_tasks": [],
									"updated_tasks": [],
									"reason": "Done"
								}`},
							}, nil
						}
						// Conclusion phase (3rd call)
						return &gollem.Response{
							Texts: []string{"Final answer"},
						}, nil
					},
					HistoryFunc: func() (*gollem.History, error) {
						return &gollem.History{}, nil
					},
				}, nil
			},
		}

		// Create strategy
		strategy := planexec.New(mockClient)
		ctx := context.Background()

		// Initialize
		err := strategy.Init(ctx, []gollem.Input{gollem.Text("Test")})
		gt.NoError(t, err)

		// Iteration 0: Planning
		state := &gollem.StrategyState{
			InitInput:    []gollem.Input{gollem.Text("Test")},
			Iteration:    0,
			Tools:        []gollem.Tool{},
			SystemPrompt: systemPrompt,
		}
		inputs, resp, err := strategy.Handle(ctx, state)
		gt.NoError(t, err)
		gt.Nil(t, resp)

		// Iteration 1: Execute task
		state.Iteration = 1
		state.NextInput = inputs
		_, _, err = strategy.Handle(ctx, state)
		gt.NoError(t, err)

		// Iteration 2: Complete task (trigger reflection and conclusion)
		state.Iteration = 2
		state.NextInput = nil
		state.LastResponse = &gollem.Response{Texts: []string{"Result"}}
		_, _, err = strategy.Handle(ctx, state)
		gt.NoError(t, err)

		// Verify conclusion was called with system prompt
		gt.Equal(t, systemPrompt, conclusionSystemPrompt)
	})
}
