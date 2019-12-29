package linear

import "fmt"

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

// NewArrayMatrix makes a new array-based Matrix with the given shape.
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

type sliceMatrix struct {
	A                        Matrix
	inLo, inHi, outLo, outHi int
}

// Slice returns a Matrix backed by another one.
func Slice(A Matrix, inLo, inHi, outLo, outHi int) Matrix {
	return &sliceMatrix{A, inLo, inHi, outLo, outHi}
}

func (s *sliceMatrix) Shape() (ins, outs int) { return s.inHi - s.inLo, s.outHi - s.outLo }
func (s *sliceMatrix) Get(in, out int) float64 {
	s.checkBounds(in, out)
	return s.A.Get(s.inLo+in, s.outLo+out)
}
func (s *sliceMatrix) Set(in, out int, value float64) {
	s.checkBounds(in, out)
	s.A.Set(s.inLo+in, s.outLo+out, value)
}
func (s *sliceMatrix) checkBounds(in, out int) {
	if in < 0 || in > s.inHi-s.inLo ||
		out < 0 || out > s.outHi-s.outLo {
		panic(fmt.Errorf("(%d, %d) is out of bounds (%d, %d)", in, out, s.inHi-s.inLo, s.outHi-s.outLo))
	}
}

// TODO: do we need a struct?
type dualMatrix struct {
	A Matrix
}

// Dual reads from a Matrix backwards, producing the Matrix transpose.
func Dual(A Matrix) Matrix {
	return &dualMatrix{A}
}

func (d *dualMatrix) Shape() (ins, outs int) {
	aIns, aOuts := d.A.Shape()
	return aOuts, aIns
}
func (d *dualMatrix) Get(in, out int) float64        { return d.A.Get(out, in) }
func (d *dualMatrix) Set(in, out int, value float64) { d.A.Set(out, in, value) }

// IsZero returns true if all of the entries are 0.
func IsZero(A Matrix) bool {
	ins, outs := A.Shape()
	for o := 0; o < outs; o++ {
		for i := 0; i < ins; i++ {
			// TODO: relax precision?
			if A.Get(i, o) != 0.0 {
				return false
			}
		}
	}
	return true
}

// CopyInto copies the entries from one matrix to another.
func CopyInto(src, dst Matrix) {
	ins, outs := src.Shape()
	dstIns, dstOuts := dst.Shape()
	if dstIns != ins || dstOuts != outs {
		panic(fmt.Errorf("dimension mismatch (%d, %d) vs (%d, %d)", ins, outs, dstIns, dstOuts))
	}
	for o := 0; o < outs; o++ {
		for i := 0; i < ins; i++ {
			dst.Set(i, o, src.Get(i, o))
		}
	}
}

// Copy produces a new Matrix with the same entries as the given one.
func Copy(A Matrix) Matrix {
	ins, outs := A.Shape()
	dst := NewArrayMatrix(ins, outs)
	CopyInto(A, dst)
	return dst
}

// IdentityInto makes the given matrix an identity.
func IdentityInto(dst Matrix) {
	ins, outs := dst.Shape()
	if ins != outs {
		panic(fmt.Errorf("dimension mismatch %d inputs vs %d outputs", ins, outs))
	}
	for o := 0; o < outs; o++ {
		for i := 0; i < ins; i++ {
			if i == o {
				dst.Set(i, o, 1)
			} else {
				dst.Set(i, o, 0)
			}
		}
	}
}

// Identity makes a new square Matrix with ones on the diagonal.
func Identity(dim int) Matrix {
	I := NewArrayMatrix(dim, dim)
	// Could use IdentityInto(I) but we only need to change the
	// diagonal.
	for i := 0; i < dim; i++ {
		I.Set(i, i, 1)
	}
	return I
}

// ComposeInto writes "A then B" (aka B*A) into dst.
func ComposeInto(A, B, dst Matrix) {
	aIns, aOuts := A.Shape()
	bIns, bOuts := B.Shape()
	if aOuts != bIns {
		panic(fmt.Errorf("dimension mismatch %d vs %d", aOuts, bIns))
	}
	for o := 0; o < bOuts; o++ {
		for i := 0; i < aIns; i++ {
			dot := 0.0
			for k := 0; k < aOuts; k++ {
				dot += A.Get(i, k) * B.Get(k, o)
			}
			dst.Set(i, o, dot)
		}
	}
}

// Compose returns "A then B" (aka B*A).
func Compose(A, B Matrix) Matrix {
	aIns, _ := A.Shape()
	_, bOuts := B.Shape()
	dst := NewArrayMatrix(aIns, bOuts)
	ComposeInto(A, B, dst)
	return dst
}

// ApplyToMatrixInto writes A*X into dst.
func ApplyToMatrixInto(A, X, dst Matrix) {
	ComposeInto(X, A, dst)
}

// ApplyToMatrix returns A*X.
func ApplyToMatrix(A, X Matrix) Matrix {
	xIns, _ := X.Shape()
	_, aOuts := A.Shape()
	dst := NewArrayMatrix(xIns, aOuts)
	ApplyToMatrixInto(A, X, dst)
	return dst
}

// ApplyToVectorInto write A*x into dst.
func ApplyToVectorInto(A Matrix, x, dst Vector) {
	ins, outs := A.Shape()
	if x.Dimension() != ins {
		panic(fmt.Errorf("dimension mismatch %d vs %d", ins, x.Dimension()))
	}
	if dst.Dimension() != outs {
		panic(fmt.Errorf("dimension mismatch %d vs %d", outs, dst.Dimension()))
	}
	for o := 0; o < outs; o++ {
		dot := 0.0
		for i := 0; i < ins; i++ {
			dot += A.Get(i, o) * x.Get(i)
		}
		dst.Set(o, dot)
	}
}

// ApplyToVector returns A*x.
func ApplyToVector(A Matrix, x Vector) Vector {
	ins, outs := A.Shape()
	if x.Dimension() != ins {
		panic(fmt.Errorf("this matrix with %d inputs cannot be applied vector with %d dimensions", ins, x.Dimension()))
	}
	dst := NewArrayVector(outs)
	ApplyToVectorInto(A, x, dst)
	return dst
}
