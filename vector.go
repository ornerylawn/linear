package linear

// Vector specifies an element of a vector space by a linear
// combination of assumed basis elements.
type Vector interface {
	// Dimension returns the size of any basis in the space, which
	// corresponds to the number of entries in the vector.
	Dimension() int
	// Get returns the scalar applied to the given basis element as part
	// of the linear combination of basis elements that the vector
	// represents.
	Get(d int) float64
	// Set changes the scalar applied to the given basis element as part
	// of the linear combination of basis elements that the vector
	// represents.
	Set(d int, value float64)
}

type arrayVector []float64

func NewArrayVector(dims int) Vector {
	return arrayVector(make([]float64, dims))
}

func (v arrayVector) Dimension() int           { return len(v) }
func (v arrayVector) Get(i int) float64        { return v[i] }
func (v arrayVector) Set(i int, value float64) { v[i] = value }
