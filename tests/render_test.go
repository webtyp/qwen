package tests

import (
	"encoding/json"
	"os"
	"testing"

	"webtyp.com/llm"
	"webtyp.com/qwen"
)

type TestCase struct {
	Name     string          `json:"name"`
	Messages []RawMessage    `json:"messages"`
	Tools    []RawToolDef    `json:"tools"`
	Prompt   string          `json:"prompt"`
}

type RawMessage struct {
	Role      string       `json:"role"`
	Content   string       `json:"content"`
	ToolCalls []RawCallDef `json:"tool_calls"`
}

type RawCallDef struct {
	Type     string          `json:"type"`
	Function RawCallFunction `json:"function"`
}

type RawCallFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type RawToolDef struct {
	Type     string          `json:"type"`
	Function RawFunctionDef `json:"function"`
}

type RawFunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

func TestRenderFixtures(t *testing.T) {
	data, err := os.ReadFile("../testdata/chat_template_cases.json")
	if err != nil {
		t.Fatalf("failed to read test fixture: %v", err)
	}

	var cases []TestCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("failed to unmarshal test cases: %v", err)
	}

	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			req := convertToLLMRequest(tc)
			segs, err := qwen.RenderForTest(req)
			if err != nil {
				t.Fatalf("render failed: %v", err)
			}

			rendered := ""
			for _, s := range segs {
				rendered += s.Text
			}

			if rendered != tc.Prompt {
				t.Errorf("rendered prompt mismatch.\nExpected:\n%q\nGot:\n%q", tc.Prompt, rendered)
			}
		})
	}
}

func convertToLLMRequest(tc TestCase) llm.Request {
	var req llm.Request

	msgStartIdx := 0
	if len(tc.Messages) > 0 && tc.Messages[0].Role == "system" {
		req.System = tc.Messages[0].Content
		msgStartIdx = 1
	}

	for i := msgStartIdx; i < len(tc.Messages); i++ {
		rm := tc.Messages[i]
		m := llm.Message{
			Content: rm.Content,
		}
		switch rm.Role {
		case "system":
			m.Role = llm.RoleSystem
		case "user":
			m.Role = llm.RoleUser
		case "assistant":
			m.Role = llm.RoleAssistant
		case "tool":
			m.Role = llm.RoleTool
		}

		for _, tc := range rm.ToolCalls {
			m.ToolCalls = append(m.ToolCalls, llm.ToolCall{
				Name:  tc.Function.Name,
				Input: string(tc.Function.Arguments),
			})
		}
		req.Messages = append(req.Messages, m)
	}

	for _, rt := range tc.Tools {
		req.Tools = append(req.Tools, llm.ToolDef{
			Name:        rt.Function.Name,
			Description: rt.Function.Description,
			InputSchema: string(rt.Function.Parameters),
		})
	}

	return req
}
