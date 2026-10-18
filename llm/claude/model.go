package claude

import "strings"

// maxOutputTokens maps a normalized Claude model ID to the maximum value the
// Messages API accepts for max_tokens on that model.
//
// Source: https://platform.claude.com/docs/en/about-claude/models/overview
// (retrieved 2026-08-10). Retired models are intentionally absent: requests to
// them fail regardless of max_tokens, so carrying a limit for them is pointless.
var maxOutputTokens = map[string]int64{
	"claude-fable-5":    128000,
	"claude-mythos-5":   128000,
	"claude-opus-5":     128000,
	"claude-opus-4-8":   128000,
	"claude-opus-4-7":   128000,
	"claude-opus-4-6":   128000,
	"claude-opus-4-5":   64000,
	"claude-sonnet-5":   128000,
	"claude-sonnet-4-6": 128000,
	"claude-sonnet-4-5": 64000,
	"claude-haiku-4-5":  64000,
}

// fallbackMaxOutputTokens is used for models absent from maxOutputTokens, such
// as a model released after this table was written or a model served through a
// compatible endpoint configured with WithBaseURL.
//
// It is the smallest limit in the table, so every currently documented model
// accepts it. A model whose real limit is lower will have the request rejected
// by the API; callers in that situation must pass WithMaxTokens explicitly.
const fallbackMaxOutputTokens int64 = 64000

// structuredOutputsUnsupported lists normalized IDs of the Claude models that
// reject output_config.format with a 400. A response schema for these models is
// sent in an instruction message instead (see responseFormatInstruction).
//
// The list names the models that lack the feature rather than the ones that
// have it, because every model released since structured outputs shipped
// supports it: a model absent from this list, including one released after it
// was written, is sent output_config.format.
//
// Sources: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
// (supported models) and https://platform.claude.com/docs/en/about-claude/models/overview
// (deprecated and retired models), retrieved 2026-09-26. Retired models are
// listed as well so that a request to them fails with the API's retirement
// error rather than with an unrelated schema error.
var structuredOutputsUnsupported = map[string]struct{}{
	"claude-sonnet-4":          {},
	"claude-sonnet-4-0":        {},
	"claude-opus-4":            {},
	"claude-opus-4-0":          {},
	"claude-3-7-sonnet":        {},
	"claude-3-7-sonnet-latest": {},
	"claude-3-5-sonnet":        {},
	"claude-3-5-sonnet-latest": {},
	"claude-3-5-sonnet-v2":     {},
	"claude-3-5-haiku":         {},
	"claude-3-5-haiku-latest":  {},
	"claude-3-opus":            {},
	"claude-3-opus-latest":     {},
	"claude-3-sonnet":          {},
	"claude-3-haiku":           {},
	"claude-2.1":               {},
	"claude-2.0":               {},
}

// supportsStructuredOutputs reports whether output_config.format can be sent to
// the given model.
func supportsStructuredOutputs(model string) bool {
	if _, ok := structuredOutputsUnsupported[model]; ok {
		return false
	}
	_, ok := structuredOutputsUnsupported[normalizeModelID(model)]
	return !ok
}

// normalizeModelID reduces the Claude API dated form, the alias form and the
// Vertex AI form of a model ID to a single key. Only an 8 digit date suffix is
// stripped, so a suffix that carries some other meaning keeps the ID distinct
// from the base model rather than silently inheriting its limit.
//
//	claude-sonnet-4-5-20250929 -> claude-sonnet-4-5
//	claude-sonnet-4-5@20250929 -> claude-sonnet-4-5
//	claude-opus-4-8            -> claude-opus-4-8
//	claude-opus-5@custom       -> claude-opus-5@custom
func normalizeModelID(model string) string {
	if i := strings.IndexByte(model, '@'); i >= 0 && isDateSuffix(model[i+1:]) {
		model = model[:i]
	}
	if i := strings.LastIndexByte(model, '-'); i >= 0 {
		if isDateSuffix(model[i+1:]) {
			model = model[:i]
		}
	}
	return model
}

// isDateSuffix reports whether s is an 8 digit date such as "20250929". Model
// IDs from the 4.6 generation onward are dateless (claude-opus-4-8), so the
// length check is what keeps their trailing segment intact.
func isDateSuffix(s string) bool {
	if len(s) != 8 {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// resolveMaxOutputTokens returns the maximum output tokens documented for the
// given model, or fallbackMaxOutputTokens when the model is not in the table.
//
// The ID is looked up verbatim before it is normalized, so a future table entry
// whose own ID ends in an 8 digit segment stays reachable.
func resolveMaxOutputTokens(model string) int64 {
	if v, ok := maxOutputTokens[model]; ok {
		return v
	}
	if v, ok := maxOutputTokens[normalizeModelID(model)]; ok {
		return v
	}
	return fallbackMaxOutputTokens
}
