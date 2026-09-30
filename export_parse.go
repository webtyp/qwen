package qwen

import "webtyp.com/llm"

// ParseToolCallsForTest exports parseToolCalls for tests in package tests.
func ParseToolCallsForTest(text string, tools []llm.ToolDef) (string, []llm.ToolCall, error) {
	return parseToolCalls(text, tools)
}
