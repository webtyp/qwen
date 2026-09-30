package qwen

func SetStepperForTest(m *Model, s stepper) {
	m.stepper = s
}

func SetVocabForTest(m *Model, vocab map[int]string) {
	maxID := 0
	for id := range vocab {
		if id > maxID {
			maxID = id
		}
	}
	vSlice := make([][]byte, maxID+1)
	for id, s := range vocab {
		vSlice[id] = []byte(s)
	}
	m.tok.vocab = vSlice
}
