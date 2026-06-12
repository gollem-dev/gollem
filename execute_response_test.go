package gollem_test

import (
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/gt"
)

func TestNewExecuteResponse(t *testing.T) {
	t.Run("create with single text", func(t *testing.T) {
		resp := gollem.NewExecuteResponse("test response")
		gt.Equal(t, []string{"test response"}, resp.Texts)
	})

	t.Run("create with multiple texts", func(t *testing.T) {
		resp := gollem.NewExecuteResponse("first", "second", "third")
		gt.Equal(t, []string{"first", "second", "third"}, resp.Texts)
	})

	t.Run("create with no texts", func(t *testing.T) {
		resp := gollem.NewExecuteResponse()
		gt.NotNil(t, resp.Texts)
		gt.Equal(t, 0, len(resp.Texts))
	})
}

func TestExecuteResponseString(t *testing.T) {
	t.Run("single text", func(t *testing.T) {
		resp := gollem.NewExecuteResponse("hello world")
		gt.Equal(t, "hello world", resp.String())
	})

	t.Run("multiple texts", func(t *testing.T) {
		resp := gollem.NewExecuteResponse("hello", "world", "test")
		gt.Equal(t, "hello world test", resp.String())
	})

	t.Run("empty texts", func(t *testing.T) {
		resp := gollem.NewExecuteResponse()
		gt.Equal(t, "", resp.String())
	})

	t.Run("nil response", func(t *testing.T) {
		var resp *gollem.ExecuteResponse
		gt.Equal(t, "", resp.String())
	})
}

func TestExecuteResponseIsEmpty(t *testing.T) {
	t.Run("nil response", func(t *testing.T) {
		var resp *gollem.ExecuteResponse
		gt.True(t, resp.IsEmpty())
	})

	t.Run("empty texts", func(t *testing.T) {
		resp := gollem.NewExecuteResponse()
		gt.True(t, resp.IsEmpty())
	})

	t.Run("single empty text", func(t *testing.T) {
		resp := gollem.NewExecuteResponse("")
		gt.True(t, resp.IsEmpty())
	})

	t.Run("non-empty texts", func(t *testing.T) {
		resp := gollem.NewExecuteResponse("test")
		gt.False(t, resp.IsEmpty())
	})

	t.Run("multiple texts", func(t *testing.T) {
		resp := gollem.NewExecuteResponse("first", "second")
		gt.False(t, resp.IsEmpty())
	})

	t.Run("multiple empty strings", func(t *testing.T) {
		resp := gollem.NewExecuteResponse("", "")
		gt.True(t, resp.IsEmpty())
	})

	t.Run("mixed empty and non-empty strings", func(t *testing.T) {
		resp := gollem.NewExecuteResponse("", "test", "")
		gt.False(t, resp.IsEmpty())
	})

	t.Run("multiple empty strings with spaces", func(t *testing.T) {
		resp := gollem.NewExecuteResponse("", "", "")
		gt.True(t, resp.IsEmpty())
		// Verify String() behavior for multiple empty strings
		gt.Equal(t, "  ", resp.String()) // Should be spaces, but IsEmpty should return true
	})
}
