package qwen

const (
	tokenEndOfText = "<|endoftext|>"
	tokenImStart   = "<|im_start|>"
	tokenImEnd     = "<|im_end|>"

	tagToolCall        = "<tool_call>"
	tagToolCallEnd     = "</tool_call>"
	tagFunctionStart   = "<function="
	tagFunctionEnd     = "</function>"
	tagParameterStart  = "<parameter="
	tagParameterEnd    = "</parameter>"
	tagToolResponse    = "<tool_response>"
	tagToolResponseEnd = "</tool_response>"

	tagThink    = "<think>"
	tagThinkEnd = "</think>"
	thinkBlock  = tagThink + "\n\n" + tagThinkEnd + "\n\n"

	toolsHeader = "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
	toolsFooter = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\n"
)
