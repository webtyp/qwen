# qwen

The Qwen3.5 language model for webtyp. It implements `llm.Client`, `llm.Streamer` and
`llm.TokenCounter` in Go, running in the browser under TinyGo, by composing `webtyp/tokenizer`,
`webtyp/weights` and `webtyp/decoder`.

## Usage

```go
import "webtyp.com/qwen"

cfg := qwen.Config{
    Decoder: qwen.Qwen35_08B,
    Weights: artifact,
}

model, err := qwen.New(cfg)
```

## Artifact Building

Build model weight artifacts using `webtyp/weightsc`:

```bash
weightsc -quant int8-block32 -prefix model.language_model.
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
