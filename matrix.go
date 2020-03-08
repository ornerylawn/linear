package linear

import (
	"fmt"
	"math"
)

// Matrix specifies a linear map under assumed bases.
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

type dualMatrix struct {
	A Matrix
}

// Dual reads from a Matrix backwards, the transpose.
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

// ApplyInto writes A*X into dst.
func ApplyInto(A, X, dst Matrix) {
	ComposeInto(X, A, dst)
}

// Apply returns A*X.
func Apply(A, X Matrix) Matrix {
	xIns, _ := X.Shape()
	_, aOuts := A.Shape()
	dst := NewArrayMatrix(xIns, aOuts)
	ApplyInto(A, X, dst)
	return dst
}

func CheckScalar(f Matrix) {
	ins, outs := f.Shape()
	if ins != 1 || outs != 1 {
		panic(fmt.Errorf("not a scalar shape=(%d, %d)", ins, outs))
	}
}

func CheckVector(v Matrix) {
	ins, outs := v.Shape()
	if ins != 1 || outs < 0 {
		panic(fmt.Errorf("not a vector shape=(%d,%d)", ins, outs))
	}
}

func CheckCovector(c Matrix) {
	ins, outs := c.Shape()
	if outs != 1 || ins < 0 {
		panic(fmt.Errorf("not a covector shape=(%d,%d)", ins, outs))
	}
}

func CheckSameIns(A, B Matrix) {
	insA, _ := A.Shape()
	insB, _ := B.Shape()
	if insA != insB {
		panic(fmt.Errorf("input dimensions don't match %d vs %d", insA, insB))
	}
}

func CheckSameOuts(A, B Matrix) {
	_, outsA := A.Shape()
	_, outsB := B.Shape()
	if outsA != outsB {
		panic(fmt.Errorf("output dimensions don't match %d vs %d", outsA, outsB))
	}
}

func CheckSameShape(A, B Matrix) {
	insA, outsA := A.Shape()
	insB, outsB := B.Shape()
	if insA != insB || outsA != outsB {
		panic(fmt.Errorf("shape mismatch (%d, %d) vs (%d, %d)", insA, outsA, insB, outsB))
	}
}

func CheckComposable(A, B Matrix) {
	_, outsA := A.Shape()
	insB, _ := B.Shape()
	if outsA != insB {
		panic(fmt.Errorf("not composable (_, %d) vs (%d, _)", outsA, insB))
	}
}

func CheckUpperTriangular(A Matrix) {
}

func CheckNotCloseToZero(x float64) {
	if math.Abs(x) < 1e-9 {
		panic(fmt.Errorf("%f is too close to zero", x))
	}
}

func DotProduct(v, c Matrix) float64 {
	CheckVector(v)
	CheckCovector(c)
	_, dim := v.Shape()
	dot := 0.0
	for d := 0; d < dim; d++ {
		dot += v.Get(0, d) * c.Get(d, 0)
	}
	return dot
}

// BasisVector make a new vector with the given dimension with a 1 in
// the given index and zeros elsewhere.
func BasisVector(dim int, index int) Matrix {
	e := NewArrayMatrix(1, dim)
	e.Set(0, index, 1)
	return e
}

// L2Norm returns the euclidean length of the vector.
func L2Norm(v Matrix) float64 {
	CheckVector(v)
	_, outs := v.Shape()
	sumOfSquares := 0.0
	for o := 0; o < outs; o++ {
		f := v.Get(0, o)
		sumOfSquares += f * f
	}
	return math.Sqrt(sumOfSquares)
}

// NormalizeInto writes into dst a vector in the same direction as src
// but with unit length, by dividing out the L2 norm.
func NormalizeInto(src, dst Matrix) {
	CheckVector(src)
	CheckVector(dst)
	CheckSameShape(src, dst)
	mag := L2Norm(src)
	_, dim := dst.Shape()
	for d := 0; d < dim; d++ {
		dst.Set(0, d, src.Get(0, d)/mag)
	}
}

// Normalize produces a vector in the same direction with unit length,
// by dividing out the L2 norm.
func Normalize(v Matrix) {
	NormalizeInto(v, v)
}
