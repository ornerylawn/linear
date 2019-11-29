package linear

// Matrix specifies a linear map under an assumed basis.
type Matrix interface {
	// Shape returns the number of inputs and outputs of the linear map,
	// which corresponds to the number of columns and rows of the
	// matrix.
	Shape() (ins, outs int)
	// Get returns the scalar applied to the given input as part of the
	// linear combination of inputs used to determine the given output,
	// which corresponds to the entry in the (in)th column and (out)th
	// row.
	Get(in, out int) float64
	// Set changes the scalar applied to the given input as part of the
	// linear combination of inputs used to determine the given output,
	// which corresponds to the entry in the (in)th column and (out)th
	// row.
	Set(in, out int, value float64)
}

type arrayMatrix struct {
	array     []float64
	ins, outs int
}

// NewArrayMatrix makes a new ArrayMatrix with the given shape.
func NewArrayMatrix(ins, outs int) Matrix {
	return &arrayMatrix{
		array: make([]float64, outs*ins),
		ins:   ins,
		outs:  outs,
	}
}

func (m *arrayMatrix) Shape() (ins, outs int)         { return m.ins, m.outs }
func (m *arrayMatrix) Get(in, out int) float64        { return m.array[out*m.ins+in] }
func (m *arrayMatrix) Set(in, out int, value float64) { m.array[out*m.ins+in] = value }
