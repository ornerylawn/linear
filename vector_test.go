package linear

import (
	"testing"
)

func TestVector(t *testing.T) {
	v := NewArrayVector(3)
	ExpectInt(3, v.Dimension(), t)
	ExpectFloat(0.0, v.Get(0), t)
	ExpectFloat(0.0, v.Get(1), t)
	ExpectFloat(0.0, v.Get(2), t)

	v.Set(1, 3.5)
	ExpectInt(3, v.Dimension(), t)
	ExpectFloat(0.0, v.Get(0), t)
	ExpectFloat(3.5, v.Get(1), t)
	ExpectFloat(0.0, v.Get(2), t)
}

func TestVectorFromColumn(t *testing.T) {
	A := NewArrayMatrix(2, 3)
	A.Set(0, 0, 1)
	A.Set(1, 0, 2)
	A.Set(0, 1, 3)
	A.Set(1, 1, 4)
	A.Set(0, 2, 5)
	A.Set(1, 2, 6)

	v := VectorFromColumn(Slice(A, 0, 1, 0, 3))

	ExpectInt(3, v.Dimension(), t)
	ExpectFloat(1, v.Get(0), t)
	ExpectFloat(3, v.Get(1), t)
	ExpectFloat(5, v.Get(2), t)

	v = VectorFromColumn(Slice(A, 1, 2, 0, 3))

	ExpectInt(3, v.Dimension(), t)
	ExpectFloat(2, v.Get(0), t)
	ExpectFloat(4, v.Get(1), t)
	ExpectFloat(6, v.Get(2), t)
}

func TestVectorFromRow(t *testing.T) {
	A := NewArrayMatrix(3, 2)
	A.Set(0, 0, 1)
	A.Set(1, 0, 2)
	A.Set(2, 0, 3)
	A.Set(0, 1, 4)
	A.Set(1, 1, 5)
	A.Set(2, 1, 6)

	v := VectorFromRow(Slice(A, 0, 3, 0, 1))

	ExpectInt(3, v.Dimension(), t)
	ExpectFloat(1, v.Get(0), t)
	ExpectFloat(2, v.Get(1), t)
	ExpectFloat(3, v.Get(2), t)

	v = VectorFromRow(Slice(A, 0, 3, 1, 2))

	ExpectInt(3, v.Dimension(), t)
	ExpectFloat(4, v.Get(0), t)
	ExpectFloat(5, v.Get(1), t)
	ExpectFloat(6, v.Get(2), t)
}

func TestBasisVector(t *testing.T) {
	e := BasisVector(5, 3)

	ExpectInt(5, e.Dimension(), t)
	for i := 0; i < e.Dimension(); i++ {
		if i == 3 {
			ExpectFloat(1, e.Get(i), t)
		} else {
			ExpectFloat(0, e.Get(i), t)
		}
	}
}

func TestL2Norm(t *testing.T) {
	v := NewArrayVector(2)
	v.Set(0, 3)
	v.Set(1, 4)

	h := L2Norm(v)

	ExpectFloat(5, h, t)
}

func TestNormalizeInto(t *testing.T) {
	v := NewArrayVector(2)
	v.Set(0, 3)
	v.Set(1, 4)

	u := NewArrayVector(2)
	NormalizeInto(v, u)

	ExpectFloat(3/5., u.Get(0), t)
	ExpectFloat(4/5., u.Get(1), t)
}

func TestNormalize(t *testing.T) {
	v := NewArrayVector(2)
	v.Set(0, 3)
	v.Set(1, 4)

	u := Normalize(v)

	ExpectInt(2, u.Dimension(), t)
	ExpectFloat(3/5., u.Get(0), t)
	ExpectFloat(4/5., u.Get(1), t)
}
