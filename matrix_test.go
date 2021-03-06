package linear

import (
	"testing"
)

func TestMatrix(t *testing.T) {
	A := NewArrayMatrix(2, 3)

	ins, outs := A.Shape()
	ExpectInt(2, ins, t)
	ExpectInt(3, outs, t)

	for o := 0; o < outs; o++ {
		for i := 0; i < ins; i++ {
			ExpectFloat(0, A.Get(i, o), t)
		}
	}

	A.Set(1, 2, 34)

	ExpectFloat(34, A.Get(1, 2), t)
}

func TestSlice(t *testing.T) {
	A := NewArrayMatrix(2, 3)
	A.Set(0, 0, 1)
	A.Set(1, 0, 2)
	A.Set(0, 1, 3)
	A.Set(1, 1, 4)
	A.Set(0, 2, 5)
	A.Set(1, 2, 6)

	S := Slice(A, 1, 2, 1, 3)

	ins, outs := S.Shape()
	ExpectInt(1, ins, t)
	ExpectInt(2, outs, t)
	ExpectFloat(4, S.Get(0, 0), t)
	ExpectFloat(6, S.Get(0, 1), t)

	A.Set(1, 1, 7)

	ExpectFloat(7, S.Get(0, 0), t)

	S.Set(0, 1, 8)

	ExpectFloat(8, A.Get(1, 2), t)
}

func TestDual(t *testing.T) {
	A := NewArrayMatrix(2, 3)
	A.Set(0, 0, 1)
	A.Set(1, 0, 2)
	A.Set(0, 1, 3)
	A.Set(1, 1, 4)
	A.Set(0, 2, 5)
	A.Set(1, 2, 6)

	B := Dual(A)

	ins, outs := B.Shape()
	ExpectInt(3, ins, t)
	ExpectInt(2, outs, t)
	ExpectFloat(1, B.Get(0, 0), t)
	ExpectFloat(2, B.Get(0, 1), t)
	ExpectFloat(3, B.Get(1, 0), t)
	ExpectFloat(4, B.Get(1, 1), t)
	ExpectFloat(5, B.Get(2, 0), t)
	ExpectFloat(6, B.Get(2, 1), t)
}

func TestIsZero(t *testing.T) {
}

func TestCopyInto(t *testing.T) {
	A := NewArrayMatrix(2, 3)
	A.Set(0, 0, 1)
	A.Set(1, 0, 2)
	A.Set(0, 1, 3)
	A.Set(1, 1, 4)
	A.Set(0, 2, 5)
	A.Set(1, 2, 6)

	B := NewArrayMatrix(2, 3)
	CopyInto(A, B)

	ExpectFloat(1, B.Get(0, 0), t)
	ExpectFloat(2, B.Get(1, 0), t)
	ExpectFloat(3, B.Get(0, 1), t)
	ExpectFloat(4, B.Get(1, 1), t)
	ExpectFloat(5, B.Get(0, 2), t)
	ExpectFloat(6, B.Get(1, 2), t)
}

func TestCopy(t *testing.T) {
	A := NewArrayMatrix(2, 3)
	A.Set(0, 0, 1)
	A.Set(1, 0, 2)
	A.Set(0, 1, 3)
	A.Set(1, 1, 4)
	A.Set(0, 2, 5)
	A.Set(1, 2, 6)

	B := Copy(A)

	ins, outs := B.Shape()
	ExpectInt(2, ins, t)
	ExpectInt(3, outs, t)
	ExpectFloat(1, B.Get(0, 0), t)
	ExpectFloat(2, B.Get(1, 0), t)
	ExpectFloat(3, B.Get(0, 1), t)
	ExpectFloat(4, B.Get(1, 1), t)
	ExpectFloat(5, B.Get(0, 2), t)
	ExpectFloat(6, B.Get(1, 2), t)
}

func TestIdentityInto(t *testing.T) {
	A := NewArrayMatrix(3, 3)
	IdentityInto(A)

	ins, outs := A.Shape()
	for o := 0; o < outs; o++ {
		for i := 0; i < ins; i++ {
			a := A.Get(i, o)
			if i == o {
				ExpectFloat(1, a, t)
			} else {
				ExpectFloat(0, a, t)
			}
		}
	}
}

func TestIdentity(t *testing.T) {
	A := Identity(3)

	ins, outs := A.Shape()
	ExpectInt(3, ins, t)
	ExpectInt(3, outs, t)
	for o := 0; o < outs; o++ {
		for i := 0; i < ins; i++ {
			a := A.Get(i, o)
			if i == o {
				ExpectFloat(1, a, t)
			} else {
				ExpectFloat(0, a, t)
			}
		}
	}
}

func TestComposeInto(t *testing.T) {
	A := NewArrayMatrix(2, 3)
	A.Set(0, 0, 2)
	A.Set(1, 0, 0)
	A.Set(0, 1, 2)
	A.Set(1, 1, 0)
	A.Set(0, 2, 0)
	A.Set(1, 2, 3)

	B := NewArrayMatrix(2, 2)
	ComposeInto(A, Dual(A), B)

	ExpectFloat(8, B.Get(0, 0), t)
	ExpectFloat(0, B.Get(1, 0), t)
	ExpectFloat(0, B.Get(0, 1), t)
	ExpectFloat(9, B.Get(1, 1), t)

	C := NewArrayMatrix(3, 3)
	ComposeInto(Dual(A), A, C)

	ExpectFloat(4, C.Get(0, 0), t)
	ExpectFloat(4, C.Get(1, 0), t)
	ExpectFloat(0, C.Get(2, 0), t)
	ExpectFloat(4, C.Get(0, 1), t)
	ExpectFloat(4, C.Get(1, 1), t)
	ExpectFloat(0, C.Get(2, 1), t)
	ExpectFloat(0, C.Get(0, 2), t)
	ExpectFloat(0, C.Get(1, 2), t)
	ExpectFloat(9, C.Get(2, 2), t)
}

func TestCompose(t *testing.T) {
	A := NewArrayMatrix(2, 3)
	A.Set(0, 0, 2)
	A.Set(1, 0, 0)
	A.Set(0, 1, 2)
	A.Set(1, 1, 0)
	A.Set(0, 2, 0)
	A.Set(1, 2, 3)

	B := Compose(A, Dual(A))

	bIns, bOuts := B.Shape()
	ExpectInt(2, bIns, t)
	ExpectInt(2, bOuts, t)
	ExpectFloat(8, B.Get(0, 0), t)
	ExpectFloat(0, B.Get(1, 0), t)
	ExpectFloat(0, B.Get(0, 1), t)
	ExpectFloat(9, B.Get(1, 1), t)

	C := Compose(Dual(A), A)

	cIns, cOuts := C.Shape()
	ExpectInt(3, cIns, t)
	ExpectInt(3, cOuts, t)
	ExpectFloat(4, C.Get(0, 0), t)
	ExpectFloat(4, C.Get(1, 0), t)
	ExpectFloat(0, C.Get(2, 0), t)
	ExpectFloat(4, C.Get(0, 1), t)
	ExpectFloat(4, C.Get(1, 1), t)
	ExpectFloat(0, C.Get(2, 1), t)
	ExpectFloat(0, C.Get(0, 2), t)
	ExpectFloat(0, C.Get(1, 2), t)
	ExpectFloat(9, C.Get(2, 2), t)
}

func TestApply(t *testing.T) {
	A := NewArrayMatrix(2, 3)
	A.Set(0, 0, 2)
	A.Set(1, 0, 0)
	A.Set(0, 1, 2)
	A.Set(1, 1, 0)
	A.Set(0, 2, 0)
	A.Set(1, 2, 3)

	B := Apply(Dual(A), A)

	bIns, bOuts := B.Shape()
	ExpectInt(2, bIns, t)
	ExpectInt(2, bOuts, t)
	ExpectFloat(8, B.Get(0, 0), t)
	ExpectFloat(0, B.Get(1, 0), t)
	ExpectFloat(0, B.Get(0, 1), t)
	ExpectFloat(9, B.Get(1, 1), t)

	C := Apply(A, Dual(A))

	cIns, cOuts := C.Shape()
	ExpectInt(3, cIns, t)
	ExpectInt(3, cOuts, t)
	ExpectFloat(4, C.Get(0, 0), t)
	ExpectFloat(4, C.Get(1, 0), t)
	ExpectFloat(0, C.Get(2, 0), t)
	ExpectFloat(4, C.Get(0, 1), t)
	ExpectFloat(4, C.Get(1, 1), t)
	ExpectFloat(0, C.Get(2, 1), t)
	ExpectFloat(0, C.Get(0, 2), t)
	ExpectFloat(0, C.Get(1, 2), t)
	ExpectFloat(9, C.Get(2, 2), t)
}

func TestApplyToVector(t *testing.T) {
	A := NewArrayMatrix(2, 2)
	A.Set(0, 0, 1)
	A.Set(1, 0, 2)
	A.Set(0, 1, 3)
	A.Set(1, 1, 4)

	x := NewArrayMatrix(1, 2)
	x.Set(0, 0, 1)
	x.Set(0, 1, 2)

	b := Apply(A, x)

	_, dim := b.Shape()
	ExpectInt(2, dim, t)
	ExpectFloat(5, b.Get(0, 0), t)
	ExpectFloat(11, b.Get(0, 1), t)
}

func TestBasisVector(t *testing.T) {
	e := BasisVector(5, 3)

	_, dim := e.Shape()
	ExpectInt(5, dim, t)
	for i := 0; i < dim; i++ {
		if i == 3 {
			ExpectFloat(1, e.Get(0, i), t)
		} else {
			ExpectFloat(0, e.Get(0, i), t)
		}
	}
}

func TestL2Norm(t *testing.T) {
	v := NewArrayMatrix(1, 2)
	v.Set(0, 0, 3)
	v.Set(0, 1, 4)

	h := L2Norm(v)

	ExpectFloat(5, h, t)
}

func TestNormalizeInto(t *testing.T) {
	v := NewArrayMatrix(1, 2)
	v.Set(0, 0, 3)
	v.Set(0, 1, 4)

	u := NewArrayMatrix(1, 2)
	NormalizeInto(v, u)

	ExpectFloat(3/5., u.Get(0, 0), t)
	ExpectFloat(4/5., u.Get(0, 1), t)
}

func TestNormalize(t *testing.T) {
	v := NewArrayMatrix(1, 2)
	v.Set(0, 0, 3)
	v.Set(0, 1, 4)

	Normalize(v)

	_, dim := v.Shape()
	ExpectInt(2, dim, t)
	ExpectFloat(3/5., v.Get(0, 0), t)
	ExpectFloat(4/5., v.Get(0, 1), t)
}
