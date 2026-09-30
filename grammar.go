package qwen

import (
	"math"

	"webtyp.com/fmt"
	"webtyp.com/llm"
)

type grammar struct {
	tools []toolSchema
	vocab [][]byte
}

func newGrammar(tools []llm.ToolDef, vocab [][]byte) *grammar {
	schemas := make([]toolSchema, len(tools))
	for i, t := range tools {
		schemas[i] = parseToolSchema(t)
	}
	return &grammar{
		tools: schemas,
		vocab: vocab,
	}
}

type grammarState struct {
	mode        int
	toolIdx     int
	seenParams  []string
	paramIdx    int
	matchBuffer string
}

const (
	gModeFreeText = iota
	gModeAfterToolCall
	gModeFunctionName
	gModeFunctionClose
	gModeParamStartOrFnEnd
	gModeParamName
	gModeParamClose
	gModeParamValue
	gModeParamEnd
	gModeToolCallEnd
)

func (g *grammar) maskLogits(logits []float32, state grammarState) {
	if len(g.tools) == 0 {
		return
	}
	minusInf := float32(math.Inf(-1))
	imStartID := lookupSpecialTokenID(tokenImStart)
	endOfTextID := lookupSpecialTokenID(tokenEndOfText)
	imEndID := lookupSpecialTokenID(tokenImEnd)
	for _, id := range [...]int{imStartID, endOfTextID} {
		if id < len(logits) {
			logits[id] = minusInf
		}
	}
	// In free text any token may follow, and <tool_call> opens the structured part: scanning
	// the 248 320-token vocabulary is only needed inside a tool call.
	if state.mode == gModeFreeText {
		return
	}
	if imEndID < len(logits) {
		logits[imEndID] = minusInf
	}
	for i := range logits {
		if i >= len(g.vocab) || i == imStartID || i == endOfTextID || i == imEndID {
			continue
		}
		if _, ok := g.stepBytes(state, g.vocab[i]); !ok {
			logits[i] = minusInf
		}
	}
}

func (g *grammar) stepBytes(st grammarState, bytes []byte) (grammarState, bool) {
	for _, b := range bytes {
		var ok bool
		st, ok = g.stepByte(st, b)
		if !ok {
			return st, false
		}
	}
	return st, true
}

func (g *grammar) stepByte(st grammarState, b byte) (grammarState, bool) {
	switch st.mode {
	case gModeFreeText:
		st.matchBuffer += string([]byte{b})
		if fmt.Contains(st.matchBuffer, tagToolCall) {
			st.mode = gModeAfterToolCall
			st.matchBuffer = ""
			return st, true
		}
		if fmt.Contains(st.matchBuffer, "<function=") ||
			fmt.Contains(st.matchBuffer, "</function>") ||
			fmt.Contains(st.matchBuffer, "<parameter=") ||
			fmt.Contains(st.matchBuffer, "</parameter>") ||
			fmt.Contains(st.matchBuffer, "</tool_call>") {
			return st, false
		}
		if len(st.matchBuffer) > 20 {
			st.matchBuffer = st.matchBuffer[len(st.matchBuffer)-15:]
		}
		return st, true

	case gModeAfterToolCall:
		target := "\n<function="
		nextChar := target[len(st.matchBuffer)]
		if b != nextChar {
			return st, false
		}
		st.matchBuffer += string([]byte{b})
		if st.matchBuffer == target {
			st.mode = gModeFunctionName
			st.matchBuffer = ""
		}
		return st, true

	case gModeFunctionName:
		st.matchBuffer += string([]byte{b})
		matchedIdx := -1
		prefixMatch := false
		for i, t := range g.tools {
			if t.name == st.matchBuffer {
				matchedIdx = i
				prefixMatch = true
				break
			}
			if fmt.HasPrefix(t.name, st.matchBuffer) {
				prefixMatch = true
			}
		}
		if matchedIdx >= 0 {
			st.toolIdx = matchedIdx
			st.mode = gModeFunctionClose
			st.matchBuffer = ""
			st.seenParams = nil
			return st, true
		}
		if prefixMatch {
			return st, true
		}
		return st, false

	case gModeFunctionClose:
		target := ">\n"
		nextChar := target[len(st.matchBuffer)]
		if b != nextChar {
			return st, false
		}
		st.matchBuffer += string([]byte{b})
		if st.matchBuffer == target {
			st.mode = gModeParamStartOrFnEnd
			st.matchBuffer = ""
		}
		return st, true

	case gModeParamStartOrFnEnd:
		if b == '<' && st.matchBuffer == "" {
			st.matchBuffer += string([]byte{b})
			return st, true
		}
		st.matchBuffer += string([]byte{b})
		pTarget := "<parameter="
		fTarget := "</function>"

		if pTarget == st.matchBuffer {
			st.mode = gModeParamName
			st.matchBuffer = ""
			return st, true
		}
		if fTarget == st.matchBuffer {
			if st.toolIdx >= 0 && st.toolIdx < len(g.tools) {
				tool := g.tools[st.toolIdx]
				for _, p := range tool.params {
					if p.required {
						hasP := false
						for _, s := range st.seenParams {
							if s == p.name {
								hasP = true
								break
							}
						}
						if !hasP {
							return st, false
						}
					}
				}
			}
			st.mode = gModeToolCallEnd
			st.matchBuffer = ""
			return st, true
		}

		if fmt.HasPrefix(pTarget, st.matchBuffer) || fmt.HasPrefix(fTarget, st.matchBuffer) {
			if fmt.HasPrefix(fTarget, st.matchBuffer) {
				if st.toolIdx >= 0 && st.toolIdx < len(g.tools) {
					tool := g.tools[st.toolIdx]
					for _, p := range tool.params {
						if p.required {
							hasP := false
							for _, s := range st.seenParams {
								if s == p.name {
									hasP = true
									break
								}
							}
							if !hasP {
								if !fmt.HasPrefix(pTarget, st.matchBuffer) {
									return st, false
								}
							}
						}
					}
				}
			}
			return st, true
		}
		return st, false

	case gModeParamName:
		st.matchBuffer += string([]byte{b})
		if st.toolIdx < 0 || st.toolIdx >= len(g.tools) {
			return st, false
		}
		tool := g.tools[st.toolIdx]

		matchedParamIdx := -1
		prefixMatch := false

		for i, p := range tool.params {
			alreadySeen := false
			for _, s := range st.seenParams {
				if s == p.name {
					alreadySeen = true
					break
				}
			}
			if alreadySeen {
				continue
			}

			if p.name == st.matchBuffer {
				matchedParamIdx = i
				prefixMatch = true
				break
			}
			if fmt.HasPrefix(p.name, st.matchBuffer) {
				prefixMatch = true
			}
		}

		if matchedParamIdx >= 0 {
			st.paramIdx = matchedParamIdx
			st.mode = gModeParamClose
			st.matchBuffer = ""
			return st, true
		}
		if prefixMatch {
			return st, true
		}
		return st, false

	case gModeParamClose:
		target := ">\n"
		nextChar := target[len(st.matchBuffer)]
		if b != nextChar {
			return st, false
		}
		st.matchBuffer += string([]byte{b})
		if st.matchBuffer == target {
			st.mode = gModeParamValue
			st.matchBuffer = ""
		}
		return st, true

	case gModeParamValue:
		if st.toolIdx < 0 || st.toolIdx >= len(g.tools) {
			return st, false
		}
		tool := g.tools[st.toolIdx]
		if st.paramIdx < 0 || st.paramIdx >= len(tool.params) {
			return st, false
		}
		param := tool.params[st.paramIdx]

		newBuf := st.matchBuffer + string([]byte{b})
		endTag := "\n</parameter>\n"

		// Check if we are matching/building the closing tag
		if fmt.HasSuffix(newBuf, endTag) {
			valStr := newBuf[:len(newBuf)-len(endTag)]
			if validateParamValue(valStr, param) {
				st.seenParams = append(st.seenParams, param.name)
				st.mode = gModeParamStartOrFnEnd
				st.matchBuffer = ""
				return st, true
			}
			return st, false
		}

		// Check if newBuf has a suffix that is a prefix of endTag
		isEndTagPrefix := false
		for k := 1; k <= len(endTag) && k <= len(newBuf); k++ {
			sub := newBuf[len(newBuf)-k:]
			if fmt.HasPrefix(endTag, sub) {
				// The value portion before 'sub'
				valPortion := newBuf[:len(newBuf)-k]
				// Check if valPortion is valid for parameter type
				if validateParamPrefix(valPortion, param) {
					isEndTagPrefix = true
					break
				}
			}
		}

		if isEndTagPrefix {
			st.matchBuffer = newBuf
			return st, true
		}

		// Not building endTag; check if entire newBuf is a valid value prefix
		if validateParamPrefix(newBuf, param) {
			st.matchBuffer = newBuf
			return st, true
		}

		return st, false

	case gModeToolCallEnd:
		st.matchBuffer += string([]byte{b})
		targetNL := "\n</tool_call>\n"

		if st.matchBuffer == targetNL {
			st.mode = gModeFreeText
			st.matchBuffer = ""
			return st, true
		}

		if fmt.HasPrefix(targetNL, st.matchBuffer) {
			return st, true
		}

		return st, false
	}

	return st, false
}

func (g *grammar) nextState(state grammarState, tokenStr string) grammarState {
	st, _ := g.stepBytes(state, []byte(tokenStr))
	return st
}
