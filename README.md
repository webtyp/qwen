# qwen
<img src="docs/img/badges.svg">

The Qwen3.5 language model for webtyp. It implements `llm.Client`, `llm.Streamer` and
`llm.TokenCounter` in Go, running in the browser under TinyGo, by composing `webtyp/tokenizer`,
`webtyp/weights` and `webtyp/decoder`.

## Usage

```go
import "webtyp.com/qwen"

art, err := weights.Open(artifactBytes) // qwen3.5-0.8b.wtypw; its tokenizer vocabulary is inside
model, err := qwen.New(qwen.Config{
    Weights: art,
    Merges:  mergesBytes, // qwen3.5-0.8b.merges
    Decoder: qwen.Qwen35_08B,
})
answer, err := model.Generate(ctx, llm.Request{System: "…", Messages: msgs, Tools: tools, MaxOutputTokens: 512})
```

`model` is also an `llm.TokenCounter` (`CountTokens`), an `llm.Streamer` (`GenerateStream`),
and an `llm.Decider` (`Decide`).
While it generates, a grammar lets the model write only an answer or well-formed tool calls
whose parameters follow each tool's schema. Special tokens typed by a person stay text.

**STATUS (remove this note when decoder v0.2.0 is published):** `webtyp/decoder` v0.1.0 reads
float32 weights only, so `New` rejects the int8 artifact until decoder v0.2.0 reads
`Int8Block32`.

## Closed questions (`llm.Decider`)

`decider-0.8b` (a Qwen3.5-0.8B fine-tune) answers closed questions by reading, in **one pass
over the prompt**, the probability of each option's letter as the next token.

`decider-0.8b` was trained on two prompt layouts:

| Layout | Route among 9 tools | Injection yes/no | Yes/no from data |
|---|---|---|---|
| **state-first** | 14/18 | **10/10** | **8/8** |
| **schema-first** | **18/18** | 6/10 | 6/8 |

**The rule:** options exactly `["no", "yes"]` → state-first; any other options → schema-first.
Schema-first places the question and options **before** the context, so a static schema or tool
list is evaluated once and cached across decisions.

```go
model, err := qwen.New(qwen.Config{
    Weights:           art,
    Merges:            mergesBytes,
    Decoder:           qwen.Qwen35_08B,
    DecideTemperature: 1.03, // decider-0.8b decision temperature
})
decision, err := model.Decide(ctx, llm.Question{
    Context: "User input or data...",
    Text:    "Which tool fits?",
    Options: []string{"search", "calculator", "none"},
})
```

## Reading only what is new

A model reads its whole prompt before answering, and that costs as much per token as writing.
`qwen` keeps the decoder state after the system block and after the last complete message, so
the next request of a conversation reads only what came after them (measured: 71.5 s for the
first turn, 21.8 s for the second). It is automatic; it helps most when the system prompt and
tools stay the same during a conversation, which `webtyp/agentcontext` guarantees.

## Artifact Building

Build the weights artifact and the merges file once, on the developer machine, with
`webtyp/weightsc` from the original checkpoint:

```bash
weightsc -in ~/Dev/LMmodels/Qwen/Qwen3.5-0.8B -out qwen3.5-0.8b.wtypw -merges-out qwen3.5-0.8b.merges \
  -id qwen3.5-0.8b -version 1 -quant int8-block32 -prefix model.language_model.
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
