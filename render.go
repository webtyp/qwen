package qwen

import (
	"webtyp.com/llm"
)

// segment represents a piece of rendered prompt: either text or a named special token.
type segment struct {
	IsSpecial bool
	Text      string
}

// render builds the Qwen prompt segments from an llm.Request.
func render(req llm.Request) ([]segment, error) {
	var segs []segment

	addSpecial := func(token string) {
		segs = append(segs, segment{IsSpecial: true, Text: token})
	}
	addText := func(text string) {
		if text != "" {
			segs = append(segs, segment{IsSpecial: false, Text: text})
		}
	}

	// 1. System block
	var sysParts []string
	if req.System != "" {
		sysParts = append(sysParts, req.System)
	}
	for _, msg := range req.Messages {
		if msg.Role == llm.RoleSystem {
			if msg.Content != "" {
				sysParts = append(sysParts, msg.Content)
			}
		}
	}

	hasTools := len(req.Tools) > 0
	hasSystem := len(sysParts) > 0

	if hasTools || hasSystem {
		addSpecial(tokenImStart)
		addText("system\n")

		if hasTools {
			addText(toolsHeader)
			for i, tool := range req.Tools {
				if i > 0 {
					addText("\n")
				}
				toolJSON := buildToolJSON(tool.Name, tool.Description, []byte(tool.InputSchema))
				addText(string(toolJSON))
			}
			addText(toolsFooter)
		}

		for i, sys := range sysParts {
			if i > 0 {
				addText("\n\n")
			}
			addText(sys)
		}

		addSpecial(tokenImEnd)
		addText("\n")
	}

	// Find index of last user message (RoleUser)
	lastUserIdx := -1
	for i, msg := range req.Messages {
		if msg.Role == llm.RoleUser {
			lastUserIdx = i
		}
	}

	// 2. Conversation turns
	inToolBlock := false

	for i := 0; i < len(req.Messages); i++ {
		msg := req.Messages[i]

		switch msg.Role {
		case llm.RoleSystem:
			// Handled in system block

		case llm.RoleUser:
			if inToolBlock {
				addSpecial(tokenImEnd)
				addText("\n")
				inToolBlock = false
			}
			addSpecial(tokenImStart)
			addText("user\n")
			addText(msg.Content)
			addSpecial(tokenImEnd)
			addText("\n")

		case llm.RoleAssistant:
			if inToolBlock {
				addSpecial(tokenImEnd)
				addText("\n")
				inToolBlock = false
			}
			addSpecial(tokenImStart)
			addText("assistant\n")

			if i > lastUserIdx {
				addText(thinkBlock)
			}

			if msg.Content != "" {
				addText(msg.Content)
			}

			if len(msg.ToolCalls) > 0 {
				for j, call := range msg.ToolCalls {
					if j > 0 {
						addText("\n")
					}
					addText(tagToolCall)
					addText("\n")
					addText(tagFunctionStart)
					addText(call.Name)
					addText(">\n")
					formattedArgs := renderCallArguments(call.Input)
					addText(formattedArgs)
					addText(tagFunctionEnd)
					addText("\n")
					addText(tagToolCallEnd)
				}
			}

			addSpecial(tokenImEnd)
			addText("\n")

		case llm.RoleTool:
			if !inToolBlock {
				addSpecial(tokenImStart)
				addText("user\n")
				inToolBlock = true
			} else {
				addText("\n")
			}
			addText(tagToolResponse)
			addText("\n")
			addText(msg.Content)
			addText("\n")
			addText(tagToolResponseEnd)
		}
	}

	if inToolBlock {
		addSpecial(tokenImEnd)
		addText("\n")
		inToolBlock = false
	}

	// Generation prompt
	addSpecial(tokenImStart)
	addText("assistant\n")
	addText(thinkBlock)

	return segs, nil
}

// renderCallArguments formats tool call input parameters into XML tags.
func renderCallArguments(input string) string {
	if input == "" {
		return ""
	}
	src := []byte(input)
	var out []byte

	// Basic JSON object scanner for key-value pairs
	i := skipWhitespace(src, 0)
	if i >= len(src) || src[i] != '{' {
		return ""
	}
	i++

	for i < len(src) {
		i = skipWhitespace(src, i)
		if i >= len(src) || src[i] == '}' {
			break
		}
		if src[i] == ',' {
			i++
			continue
		}
		if src[i] != '"' {
			break
		}

		// Read key
		key, nextI := parseJSONStringValue(src, i)
		i = skipWhitespace(src, nextI)
		if i >= len(src) || src[i] != ':' {
			break
		}
		i++ // skip ':'
		i = skipWhitespace(src, i)

		// Read value and format
		valStr, nextI := parseJSONRawValue(src, i)
		i = nextI

		out = append(out, tagParameterStart...)
		out = append(out, key...)
		out = append(out, ">\n"...)
		out = append(out, valStr...)
		out = append(out, "\n"...)
		out = append(out, tagParameterEnd...)
		out = append(out, "\n"...)
	}

	return string(out)
}

func skipWhitespace(src []byte, i int) int {
	for i < len(src) && (src[i] == ' ' || src[i] == '\t' || src[i] == '\n' || src[i] == '\r') {
		i++
	}
	return i
}

func parseJSONStringValue(src []byte, i int) (string, int) {
	if i >= len(src) || src[i] != '"' {
		return "", i
	}
	i++
	start := i
	var buf []byte
	escaped := false

	for i < len(src) {
		b := src[i]
		if escaped {
			switch b {
			case '"':
				buf = append(buf, '"')
			case '\\':
				buf = append(buf, '\\')
			case 'n':
				buf = append(buf, '\n')
			case 'r':
				buf = append(buf, '\r')
			case 't':
				buf = append(buf, '\t')
			default:
				buf = append(buf, b)
			}
			escaped = false
		} else if b == '\\' {
			escaped = true
		} else if b == '"' {
			if buf == nil {
				return string(src[start:i]), i + 1
			}
			return string(buf), i + 1
		} else {
			if buf != nil {
				buf = append(buf, b)
			}
		}
		i++
	}
	if buf != nil {
		return string(buf), i
	}
	return string(src[start:i]), i
}

func parseJSONRawValue(src []byte, i int) (string, int) {
	if i >= len(src) {
		return "", i
	}
	b := src[i]
	if b == '"' {
		return parseJSONStringValue(src, i)
	}
	if b == 't' || b == 'T' || b == 'f' || b == 'F' {
		// Boolean
		start := i
		for i < len(src) && isAlpha(src[i]) {
			i++
		}
		val := string(src[start:i])
		if val == "true" || val == "True" {
			return "True", i
		}
		if val == "false" || val == "False" {
			return "False", i
		}
		return val, i
	}
	if b == '{' || b == '[' {
		// Object or Array - reformat compact
		start := i
		depth := 0
		inStr := false
		escaped := false
		for i < len(src) {
			cb := src[i]
			if inStr {
				if escaped {
					escaped = false
				} else if cb == '\\' {
					escaped = true
				} else if cb == '"' {
					inStr = false
				}
			} else {
				if cb == '"' {
					inStr = true
				} else if cb == '{' || cb == '[' {
					depth++
				} else if cb == '}' || cb == ']' {
					depth--
					if depth == 0 {
						i++
						break
					}
				}
			}
			i++
		}
		return string(reformatJSON(src[start:i])), i
	}
	// Number or other primitive
	start := i
	for i < len(src) && (isDigit(src[i]) || src[i] == '-' || src[i] == '+' || src[i] == '.') {
		i++
	}
	return string(src[start:i]), i
}

func isAlpha(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

func isDigit(b byte) bool {
	return b >= '0' && b <= '9'
}
