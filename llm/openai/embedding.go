package openai

import (
	"context"

	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/goerr/v2"
	"github.com/sashabaranov/go-openai"
)

// GenerateEmbedding generates embeddings for the given input text.
func (c *Client) GenerateEmbedding(ctx context.Context, dimension int, input []string) ([][]float64, error) {
	/*
			AdaEmbeddingV2  EmbeddingModel = "text-embedding-ada-002"
		SmallEmbedding3 EmbeddingModel = "text-embedding-3-small"
		LargeEmbedding3 EmbeddingModel = "text-embedding-3-large"
	*/
	modelMap := map[string]openai.EmbeddingModel{
		"text-embedding-ada-002": openai.AdaEmbeddingV2,
		"text-embedding-3-small": openai.SmallEmbedding3,
		"text-embedding-3-large": openai.LargeEmbedding3,
	}

	model, ok := modelMap[c.embeddingModel]
	if !ok {
		return nil, goerr.New("invalid or unsupported embedding model. See https://platform.openai.com/docs/guides/embeddings#embedding-models", goerr.V("model", c.embeddingModel))
	}

	req := openai.EmbeddingRequest{
		Input:      input,
		Model:      model,
		Dimensions: dimension,
	}

	// Start LLM call trace span
	var traceData *trace.LLMCallData
	var llmErr error
	if h := trace.HandlerFrom(ctx); h != nil {
		ctx = h.StartLLMCall(ctx)
		defer func() { h.EndLLMCall(ctx, traceData, llmErr) }()
	}

	resp, err := c.client.CreateEmbeddings(ctx, req)
	if err != nil {
		llmErr = err
		return nil, goerr.Wrap(err, "failed to create embedding")
	}

	traceData = &trace.LLMCallData{
		InputTokens: resp.Usage.TotalTokens,
		Model:       string(resp.Model),
		Request:     &trace.LLMRequest{},
		Response:    &trace.LLMResponse{},
	}

	if len(resp.Data) == 0 {
		return nil, goerr.New("no embedding data returned")
	}

	embeddings := make([][]float64, len(resp.Data))
	for i, data := range resp.Data {
		embeddings[i] = make([]float64, len(data.Embedding))
		for j, v := range data.Embedding {
			embeddings[i][j] = float64(v)
		}
	}

	return embeddings, nil
}
