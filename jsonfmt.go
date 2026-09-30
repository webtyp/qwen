package qwen

// reformatJSON re-formats a raw JSON byte slice using ", " and ": " separators,
// leaving key order and string contents unchanged, and removing all whitespace
// outside string literals.
func reformatJSON(src []byte) []byte {
	out := make([]byte, 0, len(src))
	inString := false
	escaped := false

	for i := 0; i < len(src); i++ {
		b := src[i]

		if inString {
			out = append(out, b)
			if escaped {
				escaped = false
			} else if b == '\\' {
				escaped = true
			} else if b == '"' {
				inString = false
			}
			continue
		}

		switch b {
		case ' ', '\t', '\n', '\r':
			// skip whitespace outside strings
			continue
		case '"':
			inString = true
			out = append(out, b)
		case ':':
			out = append(out, ':', ' ')
		case ',':
			out = append(out, ',', ' ')
		default:
			out = append(out, b)
		}
	}
	return out
}

// buildToolJSON constructs the JSON representation for a single tool definition:
// {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
func buildToolJSON(name, description string, paramsJSON []byte) []byte {
	out := make([]byte, 0, 128+len(name)+len(description)+len(paramsJSON))
	out = append(out, []byte(`{"type": "function", "function": {"name": `)...)
	out = appendJSONString(out, name)
	out = append(out, []byte(`, "description": `)...)
	out = appendJSONString(out, description)
	out = append(out, []byte(`, "parameters": `)...)

	formattedParams := reformatJSON(paramsJSON)
	if len(formattedParams) == 0 {
		out = append(out, []byte("{}")...)
	} else {
		out = append(out, formattedParams...)
	}

	out = append(out, []byte("}}")...)
	return out
}

// appendJSONString appends a JSON-escaped string enclosed in double quotes to buf.
func appendJSONString(buf []byte, s string) []byte {
	buf = append(buf, '"')
	for i := 0; i < len(s); i++ {
		b := s[i]
		switch b {
		case '"':
			buf = append(buf, '\\', '"')
		case '\\':
			buf = append(buf, '\\', '\\')
		case '\n':
			buf = append(buf, '\\', 'n')
		case '\r':
			buf = append(buf, '\\', 'r')
		case '\t':
			buf = append(buf, '\\', 't')
		default:
			if b < 0x20 {
				// hex escape for control chars
				const hex = "0123456789abcdef"
				buf = append(buf, '\\', 'u', '0', '0', hex[b>>4], hex[b&0xf])
			} else {
				buf = append(buf, b)
			}
		}
	}
	buf = append(buf, '"')
	return buf
}
