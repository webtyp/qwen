package qwen

import (
	"testing"

	"webtyp.com/llm"
)

func TestParseToolCalls(t *testing.T) {
	tools := []llm.ToolDef{
		{
			Name:        "clinic_hours",
			Description: "Opening hours of the clinic for one day",
			InputSchema: `{"type": "object", "properties": {"day": {"type": "string"}}, "required": ["day"]}`,
		},
		{
			Name:        "book_appointment",
			Description: "Book an appointment",
			InputSchema: `{"type": "object", "properties": {"day": {"type": "string"}, "hour": {"type": "integer"}, "urgent": {"type": "boolean"}}, "required": ["day", "hour"]}`,
		},
	}

	t.Run("no tool calls", func(t *testing.T) {
		text := "Hola, ¿en qué puedo ayudarte?"
		content, calls, err := parseToolCalls(text, tools)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if content != "Hola, ¿en qué puedo ayudarte?" {
			t.Errorf("content mismatch: got %q", content)
		}
		if len(calls) != 0 {
			t.Errorf("expected 0 calls, got %d", len(calls))
		}
	})

	t.Run("single tool call", func(t *testing.T) {
		text := "Déjame consultar el horario.\n<tool_call>\n<function=clinic_hours>\n<parameter=day>\nlunes\n</parameter>\n</function>\n</tool_call>"
		content, calls, err := parseToolCalls(text, tools)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if content != "Déjame consultar el horario." {
			t.Errorf("content mismatch: got %q", content)
		}
		if len(calls) != 1 {
			t.Fatalf("expected 1 call, got %d", len(calls))
		}
		if calls[0].ID != "call_1" {
			t.Errorf("ID mismatch: got %q", calls[0].ID)
		}
		if calls[0].Name != "clinic_hours" {
			t.Errorf("Name mismatch: got %q", calls[0].Name)
		}
		expectedInput := `{"day": "lunes"}`
		if calls[0].Input != expectedInput {
			t.Errorf("Input mismatch: got %q, expected %q", calls[0].Input, expectedInput)
		}
	})

	t.Run("multiple tool calls and types", func(t *testing.T) {
		text := "<tool_call>\n<function=clinic_hours>\n<parameter=day>\nlunes\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=book_appointment>\n<parameter=day>\nlunes\n</parameter>\n<parameter=hour>\n9\n</parameter>\n<parameter=urgent>\nFalse\n</parameter>\n</function>\n</tool_call>"
		content, calls, err := parseToolCalls(text, tools)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if content != "" {
			t.Errorf("expected empty content, got %q", content)
		}
		if len(calls) != 2 {
			t.Fatalf("expected 2 calls, got %d", len(calls))
		}
		if calls[0].Name != "clinic_hours" || calls[0].Input != `{"day": "lunes"}` {
			t.Errorf("call 0 mismatch: %+v", calls[0])
		}
		if calls[1].Name != "book_appointment" || calls[1].Input != `{"day": "lunes", "hour": 9, "urgent": false}` {
			t.Errorf("call 1 mismatch: %+v", calls[1])
		}
	})

	t.Run("unknown tool error", func(t *testing.T) {
		text := "<tool_call>\n<function=unknown_fn>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"
		_, _, err := parseToolCalls(text, tools)
		if err == nil {
			t.Fatal("expected error for unknown tool call, got nil")
		}
	})

	t.Run("malformed xml error", func(t *testing.T) {
		text := "<tool_call>\n<function=clinic_hours>\n<parameter=day>\nlunes\n</parameter>"
		_, _, err := parseToolCalls(text, tools)
		if err == nil {
			t.Fatal("expected error for unclosed tool call, got nil")
		}
	})
}
