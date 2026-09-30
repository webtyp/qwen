package qwen

import (
	"webtyp.com/decoder"
	"webtyp.com/tokenizer"
	"webtyp.com/weights"
)

// Config configures a Qwen model instance.
type Config struct {
	Weights *weights.Artifact // from webtyp/weightsc -quant int8-block32 -prefix model.language_model.
	Merges  []byte            // the companion .merges file
	Decoder decoder.Config    // the checkpoint's shape; Qwen35_08B for the 0.8B model
}

// Qwen35_08B defines the shape/config for Qwen3.5-0.8B.
var Qwen35_08B = decoder.Config{
	Vocab:        248320,
	Hidden:       1024,
	Intermediate: 3584,
	Heads:        16,
	KVHeads:      8,
	HeadDim:      128,
	Eps:          1e-6,
}

// stepper abstracts model state stepping for generation loops and testing with fakes.
type stepper interface {
	NewState() any
	Step(state any, token int, logits []float32) error
}

// realStepper implements stepper wrapping decoder.Model and decoder.State.
type realStepper struct {
	model *decoder.Model
}

func (s *realStepper) NewState() any {
	return s.model.NewState()
}

func (s *realStepper) Step(state any, token int, logits []float32) error {
	ds := state.(*decoder.State)
	return s.model.Step(ds, token, logits)
}

// Model represents the Qwen3.5 language model.
type Model struct {
	cfg     Config
	stepper stepper
	tok     *qwenTokenizer
	bpe     *tokenizer.BPE
}

type qwenTokenizer struct {
	vocab [][]byte
}

// New creates a new Qwen model.
func New(cfg Config) (*Model, error) {
	var decModel *decoder.Model
	if cfg.Weights != nil {
		m, err := decoder.New(cfg.Decoder, cfg.Weights, "model.language_model.")
		if err != nil {
			return nil, err
		}
		decModel = m
	}

	var bpe *tokenizer.BPE
	if len(cfg.Merges) > 0 {
		b, err := tokenizer.New(tokenizer.Config{
			Scheme: tokenizer.QwenScheme{},
			Merges: cfg.Merges,
		})
		if err == nil {
			bpe = b
		}
	}

	m := &Model{
		cfg: cfg,
		tok: &qwenTokenizer{},
		bpe: bpe,
	}
	if decModel != nil {
		m.stepper = &realStepper{model: decModel}
	}

	return m, nil
}
