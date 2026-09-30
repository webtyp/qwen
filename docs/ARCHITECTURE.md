# Architecture — `webtyp/qwen`

## What this is

The adapter that turns `webtyp/decoder` into a model the agent can talk to. The agent speaks
`llm.Request` / `llm.Response`, and Qwen3.5 reads and writes text in its own format. This
repository translates between the two, and owns everything specific to the Qwen family.

You meet it in the application's composition root: `qwen.New(...)` gives the `llm.Client`,
`llm.Streamer` and `llm.TokenCounter` that `agent.Config` asks for.

## What it owns (and nothing else)

| Concern | Detail |
|---|---|
| tokenizer config | byte-level BPE, 248 320 tokens, Qwen3.5 pre-tokenizer, special tokens |
| chat template | renders `llm.Request` into Qwen's `<\|im_start\|>role … <\|im_end\|>` format: one system block (with the tool list), tool results as `<tool_response>` blocks in a user turn, empty `<think>` blocks |
| tool calls | Qwen3.5 calls tools in an **XML-like format**, not JSON: `<tool_call>` → `<function=NAME>` → `<parameter=P>` value `</parameter>`. The adapter parses it into `llm.ToolCall` with a JSON `Input` typed by the tool's schema |
| **constrained output** (v1) | the model can only produce either a final answer or a well-formed tool call; **the arguments of a tool call are constrained to that tool's JSON Schema** |
| generation | greedy/sampled next token over `decoder`, stop tokens, `MaxOutputTokens`, streaming |

## Constrained output

At every step the decoder scores all 248 320 tokens. Before choosing, this adapter masks out
every token that would break the expected format. Inside a tool call, the mask comes from the
tool's JSON Schema, so an argument the schema does not allow cannot be generated. The result is
that a 0.8B model's tool calls always parse and always type-check.

A side benefit: when the format allows exactly one continuation (the `\n<function=` after `<tool_call>`), the
expensive output projection (254 M parameters) can be skipped for that step.

## Weights

The source of truth is the original safetensors (bf16) in `~/Dev/LMmodels/Qwen/Qwen3.5-0.8B`.
`webtyp/weightsc` converts it once to **int8 in blocks of 32 values with one scale per block**
(the layout of GGUF `Q8_0`), about 750 MB for the text model. A 4-bit variant is only evaluated
if the measured speed requires it. The vision tower and the multi-token-prediction head are
not converted in v1.

The artifact is read through the ecosystem's file-reading contract. In the browser it is
implemented by `webtyp/opfs`, inside the Web Worker where the model runs.
