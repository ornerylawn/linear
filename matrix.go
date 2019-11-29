package linear

type Matrix interface {
	Shape() (ins, outs int)
	Get(in, out int) float64
	Set(in, out int, value float64)
}

type ArrayMatrix struct {
	array     []float64
	ins, outs int
}

func NewArrayMatrix(ins, outs int) Matrix {
	return &ArrayMatrix{
		array: make([]float64, outs*ins),
		ins:   ins,
		outs:  outs,
	}
}

func (m *ArrayMatrix) Shape() (ins, outs int)         { return m.ins, m.outs }
func (m *ArrayMatrix) Get(in, out int) float64        { return m.array[out*m.ins+in] }
func (m *ArrayMatrix) Set(in, out int, value float64) { m.array[out*m.ins+in] = value }
