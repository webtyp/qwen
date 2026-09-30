package qwen

import (
	"webtyp.com/fmt"
	"webtyp.com/llm"
)

// Tool schemas as the grammar needs them: parameter names, types, enums and which are required.

type paramSchema struct {
	name     string
	pType    string   // "string", "integer", "number", "boolean", "enum", "object", "array"
	enums    []string // if enum
	required bool
}

type toolSchema struct {
	name   string
	params []paramSchema
}

func parseToolSchema(td llm.ToolDef) toolSchema {
	ts := toolSchema{name: td.Name}
	schemaJSON := td.InputSchema
	if schemaJSON == "" {
		return ts
	}

	reqList := parseRequiredList(schemaJSON)
	propTypes := parseSchemaPropertyTypes(schemaJSON)

	for _, pt := range propTypes {
		ps := paramSchema{
			name:  pt.name,
			pType: pt.pType,
		}
		for _, reqName := range reqList {
			if reqName == pt.name {
				ps.required = true
				break
			}
		}
		ps.enums = parseEnumListForProperty(schemaJSON, pt.name)
		if len(ps.enums) > 0 {
			ps.pType = "enum"
		}
		ts.params = append(ts.params, ps)
	}

	return ts
}

func parseRequiredList(schemaJSON string) []string {
	var res []string
	reqIdx := fmt.Index(schemaJSON, `"required"`)
	if reqIdx == -1 {
		return res
	}
	sub := schemaJSON[reqIdx:]
	startBracket := fmt.Index(sub, "[")
	if startBracket == -1 {
		return res
	}
	sub = sub[startBracket+1:]
	endBracket := fmt.Index(sub, "]")
	if endBracket != -1 {
		sub = sub[:endBracket]
	}

	src := []byte(sub)
	i := 0
	for i < len(src) {
		i = skipWhitespace(src, i)
		if i >= len(src) {
			break
		}
		if src[i] == ',' {
			i++
			continue
		}
		if src[i] == '"' {
			val, nextI := parseJSONStringValue(src, i)
			if val != "" {
				res = append(res, val)
			}
			i = nextI
		} else {
			i++
		}
	}
	return res
}

func parseEnumListForProperty(schemaJSON string, propName string) []string {
	var res []string
	pIdx := fmt.Index(schemaJSON, `"`+propName+`"`)
	if pIdx == -1 {
		return res
	}
	sub := schemaJSON[pIdx:]
	enumIdx := fmt.Index(sub, `"enum"`)
	if enumIdx == -1 {
		return res
	}
	sub = sub[enumIdx:]
	startBracket := fmt.Index(sub, "[")
	if startBracket == -1 {
		return res
	}
	sub = sub[startBracket+1:]
	endBracket := fmt.Index(sub, "]")
	if endBracket != -1 {
		sub = sub[:endBracket]
	}

	src := []byte(sub)
	i := 0
	for i < len(src) {
		i = skipWhitespace(src, i)
		if i >= len(src) {
			break
		}
		if src[i] == ',' {
			i++
			continue
		}
		if src[i] == '"' {
			val, nextI := parseJSONStringValue(src, i)
			res = append(res, val)
			i = nextI
		} else {
			i++
		}
	}
	return res
}

func validateParamPrefix(val string, param paramSchema) bool {
	switch param.pType {
	case "enum":
		if val == "" {
			return true
		}
		for _, e := range param.enums {
			if fmt.HasPrefix(e, val) {
				return true
			}
		}
		return false
	case "integer":
		if val == "" {
			return true
		}
		for i := 0; i < len(val); i++ {
			if i == 0 && val[i] == '-' {
				continue
			}
			if val[i] < '0' || val[i] > '9' {
				return false
			}
		}
		return true
	case "number":
		if val == "" {
			return true
		}
		hasDot := false
		for i := 0; i < len(val); i++ {
			if i == 0 && val[i] == '-' {
				continue
			}
			if val[i] == '.' {
				if hasDot {
					return false
				}
				hasDot = true
				continue
			}
			if val[i] < '0' || val[i] > '9' {
				return false
			}
		}
		return true
	case "boolean":
		if val == "" {
			return true
		}
		return fmt.HasPrefix("true", val) || fmt.HasPrefix("false", val) || fmt.HasPrefix("True", val) || fmt.HasPrefix("False", val)
	default: // string, object, array
		// Cannot contain tags
		if fmt.Contains(val, "<tool_call>") ||
			fmt.Contains(val, "</tool_call>") ||
			fmt.Contains(val, "<function=") ||
			fmt.Contains(val, "</function>") ||
			fmt.Contains(val, "<parameter=") {
			return false
		}
		return true
	}
}

func validateParamValue(val string, param paramSchema) bool {
	switch param.pType {
	case "integer":
		if len(val) == 0 {
			return false
		}
		for i := 0; i < len(val); i++ {
			if i == 0 && val[i] == '-' {
				continue
			}
			if val[i] < '0' || val[i] > '9' {
				return false
			}
		}
		return true
	case "number":
		if len(val) == 0 {
			return false
		}
		hasDot := false
		for i := 0; i < len(val); i++ {
			if i == 0 && val[i] == '-' {
				continue
			}
			if val[i] == '.' {
				if hasDot {
					return false
				}
				hasDot = true
				continue
			}
			if val[i] < '0' || val[i] > '9' {
				return false
			}
		}
		return true
	case "boolean":
		return val == "true" || val == "false" || val == "True" || val == "False"
	case "enum":
		for _, e := range param.enums {
			if e == val {
				return true
			}
		}
		return false
	default: // string, object, array
		return true
	}
}
