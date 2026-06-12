package gollem_test

import (
	"errors"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/gt"
)

func TestParameterValidation(t *testing.T) {
	t.Run("number constraints", func(t *testing.T) {
		t.Run("valid minimum and maximum", func(t *testing.T) {
			p := &gollem.Parameter{
				Type:    gollem.TypeNumber,
				Minimum: ptr(1.0),
				Maximum: ptr(10.0),
			}
			gt.NoError(t, p.Validate())
		})

		t.Run("invalid minimum and maximum", func(t *testing.T) {
			p := &gollem.Parameter{
				Type:    gollem.TypeNumber,
				Minimum: ptr(10.0),
				Maximum: ptr(1.0),
			}
			gt.Error(t, p.Validate())
		})
	})

	t.Run("string constraints", func(t *testing.T) {
		t.Run("valid minLength and maxLength", func(t *testing.T) {
			p := &gollem.Parameter{
				Type:      gollem.TypeString,
				MinLength: ptr(1),
				MaxLength: ptr(10),
			}
			gt.NoError(t, p.Validate())
		})

		t.Run("invalid minLength and maxLength", func(t *testing.T) {
			p := &gollem.Parameter{
				Type:      gollem.TypeString,
				MinLength: ptr(10),
				MaxLength: ptr(1),
			}
			gt.Error(t, p.Validate())
		})

		t.Run("valid pattern", func(t *testing.T) {
			p := &gollem.Parameter{
				Type:    gollem.TypeString,
				Pattern: "^[a-z]+$",
			}
			gt.NoError(t, p.Validate())
		})

		t.Run("invalid pattern", func(t *testing.T) {
			p := &gollem.Parameter{
				Type:    gollem.TypeString,
				Pattern: "[invalid",
			}
			gt.Error(t, p.Validate())
		})
	})

	t.Run("array constraints", func(t *testing.T) {
		t.Run("valid minItems and maxItems", func(t *testing.T) {
			p := &gollem.Parameter{
				Type:     gollem.TypeArray,
				Items:    &gollem.Parameter{Type: gollem.TypeString},
				MinItems: ptr(1),
				MaxItems: ptr(10),
			}
			gt.NoError(t, p.Validate())
		})

		t.Run("invalid minItems and maxItems", func(t *testing.T) {
			p := &gollem.Parameter{
				Type:     gollem.TypeArray,
				Items:    &gollem.Parameter{Type: gollem.TypeString},
				MinItems: ptr(10),
				MaxItems: ptr(1),
			}
			gt.Error(t, p.Validate())
		})
	})

	t.Run("object constraints", func(t *testing.T) {
		t.Run("valid properties", func(t *testing.T) {
			p := &gollem.Parameter{
				Type: gollem.TypeObject,
				Properties: map[string]*gollem.Parameter{
					"name": {
						Type:        gollem.TypeString,
						Description: "User name",
					},
					"age": {
						Type:        gollem.TypeNumber,
						Description: "User age",
					},
				},
			}
			gt.NoError(t, p.Validate())
		})

		t.Run("duplicate property names", func(t *testing.T) {
			p := &gollem.Parameter{
				Type:       gollem.TypeObject,
				Properties: make(map[string]*gollem.Parameter),
			}
			p.Properties["name"] = &gollem.Parameter{
				Type:        gollem.TypeString,
				Description: "User name",
			}
			p.Properties["name"] = &gollem.Parameter{
				Type:        gollem.TypeString,
				Description: "Duplicate name",
			}
			gt.NoError(t, p.Validate())
		})

		t.Run("invalid property type", func(t *testing.T) {
			p := &gollem.Parameter{
				Type: gollem.TypeObject,
				Properties: map[string]*gollem.Parameter{
					"name": {
						Type:        "invalid",
						Description: "User name",
					},
				},
			}
			gt.Error(t, p.Validate())
		})
	})
}

func ptr[T any](v T) *T {
	return &v
}

func TestToolSpecValidation(t *testing.T) {
	t.Run("valid tool spec", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name:        "test",
			Description: "test description",
			Parameters: map[string]*gollem.Parameter{
				"param1": {
					Type:        gollem.TypeString,
					Description: "test parameter",
					Required:    true,
				},
			},
		}
		gt.NoError(t, spec.Validate())
	})

	t.Run("empty name", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Description: "test description",
			Parameters: map[string]*gollem.Parameter{
				"param1": {
					Type:        gollem.TypeString,
					Description: "test parameter",
				},
			},
		}
		gt.Error(t, spec.Validate())
	})

	t.Run("invalid parameter type", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name:        "test",
			Description: "test description",
			Parameters: map[string]*gollem.Parameter{
				"param1": {
					Type:        "invalid",
					Description: "test parameter",
				},
			},
		}
		gt.Error(t, spec.Validate())
	})

	t.Run("invalid parameter", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name:        "test",
			Description: "test description",
			Parameters: map[string]*gollem.Parameter{
				"param1": {
					Type:    gollem.TypeNumber,
					Minimum: ptr(10.0),
					Maximum: ptr(1.0),
				},
			},
		}
		gt.Error(t, spec.Validate())
	})

	t.Run("object parameter without properties", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name:        "test",
			Description: "test description",
			Parameters: map[string]*gollem.Parameter{
				"param1": {
					Type:        gollem.TypeObject,
					Description: "test parameter",
				},
			},
		}
		gt.Error(t, spec.Validate())
	})

	t.Run("array parameter without items", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name:        "test",
			Description: "test description",
			Parameters: map[string]*gollem.Parameter{
				"param1": {
					Type:        gollem.TypeArray,
					Description: "test parameter",
				},
			},
		}
		gt.Error(t, spec.Validate())
	})
}

func TestToolSpecValidateArgs(t *testing.T) {
	t.Run("all valid args", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name:        "search",
			Description: "search tool",
			Parameters: map[string]*gollem.Parameter{
				"query":       {Type: gollem.TypeString, Required: true},
				"max_results": {Type: gollem.TypeInteger},
			},
		}
		err := spec.ValidateArgs(map[string]any{
			"query":       "hello",
			"max_results": 10,
		})
		gt.NoError(t, err)
	})

	t.Run("missing required parameter", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name: "search",
			Parameters: map[string]*gollem.Parameter{
				"query": {Type: gollem.TypeString, Required: true},
			},
		}
		err := spec.ValidateArgs(map[string]any{})
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrToolArgsValidation))
	})

	t.Run("wrong type parameter", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name: "search",
			Parameters: map[string]*gollem.Parameter{
				"count": {Type: gollem.TypeInteger},
			},
		}
		err := spec.ValidateArgs(map[string]any{"count": "not a number"})
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrToolArgsValidation))
	})

	t.Run("multiple validation errors collected", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name: "search",
			Parameters: map[string]*gollem.Parameter{
				"query":  {Type: gollem.TypeString, Required: true},
				"count":  {Type: gollem.TypeInteger},
				"format": {Type: gollem.TypeString, Enum: []string{"json", "csv"}},
			},
		}
		err := spec.ValidateArgs(map[string]any{
			"count":  "not a number",
			"format": "xml",
		})
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrToolArgsValidation))
		// Error message should mention tool name
		gt.S(t, err.Error()).Contains("search")
	})

	t.Run("nil parameters is valid", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name:       "noop",
			Parameters: nil,
		}
		err := spec.ValidateArgs(map[string]any{})
		gt.NoError(t, err)
	})

	t.Run("optional parameter with nil value is valid", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name: "search",
			Parameters: map[string]*gollem.Parameter{
				"query": {Type: gollem.TypeString, Required: false},
			},
		}
		err := spec.ValidateArgs(map[string]any{})
		gt.NoError(t, err)
	})

	t.Run("error message contains structured info", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name: "my_tool",
			Parameters: map[string]*gollem.Parameter{
				"name": {Type: gollem.TypeString, Required: true},
			},
		}
		err := spec.ValidateArgs(map[string]any{})
		gt.Error(t, err)
		msg := err.Error()
		gt.S(t, msg).Contains("my_tool")
		gt.S(t, msg).Contains("Please correct the arguments and retry the tool call.")
	})

	t.Run("nil args map treats all values as nil", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name: "search",
			Parameters: map[string]*gollem.Parameter{
				"query": {Type: gollem.TypeString, Required: true},
			},
		}
		err := spec.ValidateArgs(nil)
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrToolArgsValidation))
	})

	t.Run("nil args map with optional params is valid", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name: "search",
			Parameters: map[string]*gollem.Parameter{
				"query": {Type: gollem.TypeString, Required: false},
			},
		}
		err := spec.ValidateArgs(nil)
		gt.NoError(t, err)
	})

	t.Run("nested object property validation", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name: "create_user",
			Parameters: map[string]*gollem.Parameter{
				"user": {
					Type: gollem.TypeObject,
					Properties: map[string]*gollem.Parameter{
						"name": {Type: gollem.TypeString, Required: true},
						"age":  {Type: gollem.TypeInteger},
					},
				},
			},
		}

		// Valid nested object
		gt.NoError(t, spec.ValidateArgs(map[string]any{
			"user": map[string]any{"name": "Alice", "age": 30},
		}))

		// Missing required nested property
		err := spec.ValidateArgs(map[string]any{
			"user": map[string]any{"age": 30},
		})
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrToolArgsValidation))

		// Wrong type in nested property
		err = spec.ValidateArgs(map[string]any{
			"user": map[string]any{"name": 123},
		})
		gt.Error(t, err)
	})

	t.Run("array element validation", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name: "tag_items",
			Parameters: map[string]*gollem.Parameter{
				"tags": {
					Type:  gollem.TypeArray,
					Items: &gollem.Parameter{Type: gollem.TypeString},
				},
			},
		}

		// Valid array
		gt.NoError(t, spec.ValidateArgs(map[string]any{
			"tags": []any{"go", "rust"},
		}))

		// Invalid element type in array
		err := spec.ValidateArgs(map[string]any{
			"tags": []any{"go", 123},
		})
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrToolArgsValidation))
	})

	t.Run("number constraint violation", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name: "set_volume",
			Parameters: map[string]*gollem.Parameter{
				"level": {Type: gollem.TypeNumber, Minimum: ptr(0.0), Maximum: ptr(100.0)},
			},
		}

		gt.NoError(t, spec.ValidateArgs(map[string]any{"level": 50.0}))

		err := spec.ValidateArgs(map[string]any{"level": 200.0})
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrToolArgsValidation))

		err = spec.ValidateArgs(map[string]any{"level": -1.0})
		gt.Error(t, err)
	})

	t.Run("string enum constraint violation", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name: "set_mode",
			Parameters: map[string]*gollem.Parameter{
				"mode": {Type: gollem.TypeString, Enum: []string{"auto", "manual"}},
			},
		}

		gt.NoError(t, spec.ValidateArgs(map[string]any{"mode": "auto"}))

		err := spec.ValidateArgs(map[string]any{"mode": "turbo"})
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrToolArgsValidation))
	})

	t.Run("string pattern constraint violation", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name: "set_id",
			Parameters: map[string]*gollem.Parameter{
				"id": {Type: gollem.TypeString, Pattern: "^[A-Z]{3}-[0-9]{4}$"},
			},
		}

		gt.NoError(t, spec.ValidateArgs(map[string]any{"id": "ABC-1234"}))

		err := spec.ValidateArgs(map[string]any{"id": "invalid"})
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrToolArgsValidation))
	})

	t.Run("extra args not in spec are ignored", func(t *testing.T) {
		spec := gollem.ToolSpec{
			Name: "search",
			Parameters: map[string]*gollem.Parameter{
				"query": {Type: gollem.TypeString, Required: true},
			},
		}
		err := spec.ValidateArgs(map[string]any{
			"query":          "hello",
			"unknown_param":  "extra",
			"another_unused": 42,
		})
		gt.NoError(t, err)
	})
}

func TestValidateValue(t *testing.T) {
	t.Run("required parameter", func(t *testing.T) {
		t.Run("nil value returns error", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeString, Required: true}
			gt.Error(t, p.ValidateValue("test", nil))
		})

		t.Run("valid value passes", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeString, Required: true}
			gt.NoError(t, p.ValidateValue("test", "hello"))
		})
	})

	t.Run("optional parameter", func(t *testing.T) {
		t.Run("nil value is valid", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeString, Required: false}
			gt.NoError(t, p.ValidateValue("test", nil))
		})
	})

	t.Run("string type", func(t *testing.T) {
		t.Run("valid string", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeString}
			gt.NoError(t, p.ValidateValue("test", "hello"))
		})

		t.Run("non-string returns error", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeString}
			gt.Error(t, p.ValidateValue("test", 123))
		})

		t.Run("enum validation passes", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeString, Enum: []string{"a", "b", "c"}}
			gt.NoError(t, p.ValidateValue("test", "b"))
		})

		t.Run("enum validation fails", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeString, Enum: []string{"a", "b", "c"}}
			gt.Error(t, p.ValidateValue("test", "d"))
		})

		t.Run("minLength validation", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeString, MinLength: ptr(3)}
			gt.Error(t, p.ValidateValue("test", "ab"))
			gt.NoError(t, p.ValidateValue("test", "abc"))
		})

		t.Run("maxLength validation", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeString, MaxLength: ptr(3)}
			gt.NoError(t, p.ValidateValue("test", "abc"))
			gt.Error(t, p.ValidateValue("test", "abcd"))
		})

		t.Run("pattern validation", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeString, Pattern: "^[a-z]+$"}
			gt.NoError(t, p.ValidateValue("test", "abc"))
			gt.Error(t, p.ValidateValue("test", "ABC"))
		})
	})

	t.Run("number type", func(t *testing.T) {
		t.Run("valid float64", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeNumber}
			gt.NoError(t, p.ValidateValue("test", 3.14))
		})

		t.Run("valid int", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeNumber}
			gt.NoError(t, p.ValidateValue("test", 42))
		})

		t.Run("non-number returns error", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeNumber}
			gt.Error(t, p.ValidateValue("test", "not a number"))
		})

		t.Run("minimum validation", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeNumber, Minimum: ptr(5.0)}
			gt.Error(t, p.ValidateValue("test", 4.0))
			gt.NoError(t, p.ValidateValue("test", 5.0))
		})

		t.Run("maximum validation", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeNumber, Maximum: ptr(10.0)}
			gt.NoError(t, p.ValidateValue("test", 10.0))
			gt.Error(t, p.ValidateValue("test", 11.0))
		})
	})

	t.Run("integer type", func(t *testing.T) {
		t.Run("valid int", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeInteger}
			gt.NoError(t, p.ValidateValue("test", 42))
		})

		t.Run("float64 that is integer is valid", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeInteger}
			gt.NoError(t, p.ValidateValue("test", float64(42)))
		})

		t.Run("non-integer float64 returns error", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeInteger}
			gt.Error(t, p.ValidateValue("test", 3.14))
		})
	})

	t.Run("boolean type", func(t *testing.T) {
		t.Run("valid bool", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeBoolean}
			gt.NoError(t, p.ValidateValue("test", true))
			gt.NoError(t, p.ValidateValue("test", false))
		})

		t.Run("non-bool returns error", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeBoolean}
			gt.Error(t, p.ValidateValue("test", "true"))
		})
	})

	t.Run("array type", func(t *testing.T) {
		t.Run("valid array", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeArray, Items: &gollem.Parameter{Type: gollem.TypeString}}
			gt.NoError(t, p.ValidateValue("test", []any{"a", "b", "c"}))
		})

		t.Run("non-array returns error", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeArray, Items: &gollem.Parameter{Type: gollem.TypeString}}
			gt.Error(t, p.ValidateValue("test", "not an array"))
		})

		t.Run("minItems validation", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeArray, Items: &gollem.Parameter{Type: gollem.TypeString}, MinItems: ptr(2)}
			gt.Error(t, p.ValidateValue("test", []any{"a"}))
			gt.NoError(t, p.ValidateValue("test", []any{"a", "b"}))
		})

		t.Run("maxItems validation", func(t *testing.T) {
			p := &gollem.Parameter{Type: gollem.TypeArray, Items: &gollem.Parameter{Type: gollem.TypeString}, MaxItems: ptr(2)}
			gt.NoError(t, p.ValidateValue("test", []any{"a", "b"}))
			gt.Error(t, p.ValidateValue("test", []any{"a", "b", "c"}))
		})
	})

	t.Run("object type", func(t *testing.T) {
		t.Run("valid object", func(t *testing.T) {
			p := &gollem.Parameter{
				Type: gollem.TypeObject,
				Properties: map[string]*gollem.Parameter{
					"name": {Type: gollem.TypeString, Required: true},
					"age":  {Type: gollem.TypeNumber},
				},
			}
			gt.NoError(t, p.ValidateValue("test", map[string]any{"name": "Alice", "age": 30.0}))
		})

		t.Run("non-object returns error", func(t *testing.T) {
			p := &gollem.Parameter{
				Type: gollem.TypeObject,
				Properties: map[string]*gollem.Parameter{
					"name": {Type: gollem.TypeString},
				},
			}
			gt.Error(t, p.ValidateValue("test", "not an object"))
		})

		t.Run("missing required property returns error", func(t *testing.T) {
			p := &gollem.Parameter{
				Type: gollem.TypeObject,
				Properties: map[string]*gollem.Parameter{
					"name": {Type: gollem.TypeString, Required: true},
				},
			}
			gt.Error(t, p.ValidateValue("test", map[string]any{}))
		})

		t.Run("invalid property type returns error", func(t *testing.T) {
			p := &gollem.Parameter{
				Type: gollem.TypeObject,
				Properties: map[string]*gollem.Parameter{
					"age": {Type: gollem.TypeNumber},
				},
			}
			gt.Error(t, p.ValidateValue("test", map[string]any{"age": "not a number"}))
		})
	})
}
