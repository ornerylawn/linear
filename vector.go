package linear

import (
	"fmt"
	"math"
)

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

// NewArrayVector makes a new array-based Vector with the given
// dimension.
func NewArrayVector(dims int) Vector {
	return arrayVector(make([]float64, dims))
}

func (v arrayVector) Dimension() int           { return len(v) }
func (v arrayVector) Get(i int) float64        { return v[i] }
func (v arrayVector) Set(i int, value float64) { v[i] = value }

type sliceColumn struct {
	A Matrix
}

// VectorFromColumn creates a Vector that reads from a Matrix that has
// only one column.
func VectorFromColumn(A Matrix) Vector {
	ins, _ := A.Shape()
	if ins != 1 {
		panic(fmt.Errorf("can't make vector from matrix that has %d columns", ins))
	}
	return &sliceColumn{A}
}

func (s *sliceColumn) Dimension() int {
	_, outs := s.A.Shape()
	return outs
}
func (s *sliceColumn) Get(d int) float64        { return s.A.Get(0, d) }
func (s *sliceColumn) Set(d int, value float64) { s.A.Set(0, d, value) }

type sliceRow struct {
	A Matrix
}

// VectorFromRow creates a Vector that reads from a Matrix that has
// only one row.
func VectorFromRow(A Matrix) Vector {
	_, outs := A.Shape()
	if outs != 1 {
		panic(fmt.Errorf("can't make vector from matrix that has %d rows", outs))
	}
	return &sliceRow{A}
}

func (s *sliceRow) Dimension() int {
	ins, _ := s.A.Shape()
	return ins
}
func (s *sliceRow) Get(d int) float64        { return s.A.Get(d, 0) }
func (s *sliceRow) Set(d int, value float64) { s.A.Set(d, 0, value) }

// BasisVector make a new vector with the given dimension with a 1 in
// the given index and zeros elsewhere.
func BasisVector(dim int, index int) Vector {
	e := NewArrayVector(dim)
	e.Set(index, 1)
	return e
}

// L2Norm returns the euclidean length of the vector.
func L2Norm(v Vector) float64 {
	dims := v.Dimension()
	sumOfSquares := 0.0
	for d := 0; d < dims; d++ {
		a := v.Get(d)
		sumOfSquares += a * a
	}
	return math.Sqrt(sumOfSquares)
}

// NormalizeInto writes into dst a vector in the same direction as src
// but with unit length, by dividing out the L2 norm.
func NormalizeInto(src, dst Vector) {
	dim := src.Dimension()
	if dst.Dimension() != dim {
		panic(fmt.Errorf("dimension mismatch %d vs %d", dim, dst.Dimension()))
	}
	mag := L2Norm(src)
	for d := 0; d < dim; d++ {
		dst.Set(d, src.Get(d)/mag)
	}
}

// Normalize produces a vector in the same direction with unit length,
// by dividing out the L2 norm.
func Normalize(v Vector) Vector {
	dim := v.Dimension()
	dst := NewArrayVector(dim)
	NormalizeInto(v, dst)
	return dst
}
