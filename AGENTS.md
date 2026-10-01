# AGENTS.md — Instructions for AI Agents in `webtyp.com/qwen`

## Root package tests and reasons

Per project design rules, tests in this repository live in the root package (`package qwen`) for the following reasons:

- `render_test.go`: exercises prompt template segment rendering against `testdata/chat_template_cases.json`.
- `parse_test.go`: exercises XML tool call parser implementation (`parseToolCalls`, `parseSingleToolCallBlock`).
- `grammar_test.go`: exercises character-level state machine logit masking (`maskLogits`, `stepBytes`, required parameter enforcement, enums, integers, and 1,000 random forced seeds).
- `generate_test.go`: exercises model generation loops with fake scripted steppers.
- `cache_test.go`: exercises the prefix cache through the unexported stepper (ids fed per request).
- `injection_test.go`: exercises prompt injection safety during segment generation.

These tests observe unexported template rendering, XML parsing, and grammar state machine functions directly to ensure strict compliance with Qwen3.5 format constraints.
