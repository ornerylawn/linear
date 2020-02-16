// TODO: need to test non-square cases.
// TODO: test numerical stability.
// TODO: benchmark/optimize.

package linear

import (
	"math"
	"math/rand"
	"testing"
)

func ExpectFloat(expect, got float64, t *testing.T) {
	if math.Abs(got-expect) > 1e-9 {
		t.Errorf("expected %f but got %f", expect, got)
	}
}

func ExpectInt(expect, got int, t *testing.T) {
	if got != expect {
		t.Errorf("expected %d but got %d", expect, got)
	}
}

func TestFindInputToUpperTriangular(t *testing.T) {
	A := NewArrayMatrix(3, 3)
	A.Set(0, 0, 1)
	A.Set(1, 0, 2)
	A.Set(2, 0, 3)
	A.Set(0, 1, 0)
	A.Set(1, 1, 4)
	A.Set(2, 1, 5)
	A.Set(0, 2, 0)
	A.Set(1, 2, 0)
	A.Set(2, 2, 6)

	b := NewArrayVector(3)
	b.Set(0, 1)
	b.Set(1, 2)
	b.Set(2, 3)

	x := FindInputToUpperTriangular(A, b)

	ExpectInt(3, x.Dimension(), t)
	ExpectFloat(-1.0/4.0, x.Get(0), t)
	ExpectFloat(-1.0/8.0, x.Get(1), t)
	ExpectFloat(1.0/2.0, x.Get(2), t)
}

func TestHouseholderInto(t *testing.T) {
	A0 := NewArrayMatrix(3, 3)
	A0.Set(0, 0, 12)
	A0.Set(1, 0, -51)
	A0.Set(2, 0, 4)
	A0.Set(0, 1, 6)
	A0.Set(1, 1, 167)
	A0.Set(2, 1, -68)
	A0.Set(0, 2, -4)
	A0.Set(1, 2, 24)
	A0.Set(2, 2, -41)

	H0 := NewArrayMatrix(3, 3)
	HouseholderInto(
		VectorFromColumn(Slice(A0, 0, 1, 0, 3)),
		BasisVector(3, 0),
		H0)

	h0Ins, h0Outs := H0.Shape()
	ExpectInt(3, h0Ins, t)
	ExpectInt(3, h0Outs, t)
	ExpectFloat(-0.857142857142857, H0.Get(0, 0), t)
	ExpectFloat(-0.42857142857142855, H0.Get(1, 0), t)
	ExpectFloat(0.2857142857142857, H0.Get(2, 0), t)
	ExpectFloat(-0.42857142857142855, H0.Get(0, 1), t)
	ExpectFloat(0.9010989010989011, H0.Get(1, 1), t)
	ExpectFloat(0.06593406593406594, H0.Get(2, 1), t)
	ExpectFloat(0.2857142857142857, H0.Get(0, 2), t)
	ExpectFloat(0.06593406593406594, H0.Get(1, 2), t)
	ExpectFloat(0.9560439560439561, H0.Get(2, 2), t)

	A1 := ApplyToMatrix(H0, A0)

	a1Ins, a1Outs := A1.Shape()
	ExpectInt(3, a1Ins, t)
	ExpectInt(3, a1Outs, t)
	ExpectFloat(-13.999999999999998, A1.Get(0, 0), t)
	ExpectFloat(-21.000000000000004, A1.Get(1, 0), t)
	ExpectFloat(14.000000000000002, A1.Get(2, 0), t)
	ExpectFloat(8.326672684688674e-16, A1.Get(0, 1), t)
	ExpectFloat(173.92307692307693, A1.Get(1, 1), t)
	ExpectFloat(-65.6923076923077, A1.Get(2, 1), t)
	ExpectFloat(-4.440892098500626e-16, A1.Get(0, 2), t)
	ExpectFloat(19.384615384615387, A1.Get(1, 2), t)
	ExpectFloat(-42.53846153846154, A1.Get(2, 2), t)

	H1 := Identity(3)
	HouseholderInto(
		VectorFromColumn(Slice(A1, 1, 2, 1, 3)),
		BasisVector(2, 0),
		Slice(H1, 1, 3, 1, 3))

	h1Ins, h1Outs := H1.Shape()
	ExpectInt(3, h1Ins, t)
	ExpectInt(3, h1Outs, t)
	ExpectFloat(1.0, H1.Get(0, 0), t)
	ExpectFloat(0.0, H1.Get(1, 0), t)
	ExpectFloat(0.0, H1.Get(2, 0), t)
	ExpectFloat(0.0, H1.Get(0, 1), t)
	ExpectFloat(-0.9938461538461536, H1.Get(1, 1), t)
	ExpectFloat(-0.11076923076923079, H1.Get(2, 1), t)
	ExpectFloat(0.0, H1.Get(0, 2), t)
	ExpectFloat(-0.11076923076923079, H1.Get(1, 2), t)
	ExpectFloat(0.9938461538461538, H1.Get(2, 2), t)

	A2 := ApplyToMatrix(H1, A1)

	a2ins, a2outs := A2.Shape()
	ExpectInt(3, a2ins, t)
	ExpectInt(3, a2outs, t)
	ExpectFloat(-13.999999999999998, A2.Get(0, 0), t)
	ExpectFloat(-21.000000000000004, A2.Get(1, 0), t)
	ExpectFloat(14.000000000000002, A2.Get(2, 0), t)
	ExpectFloat(-7.783517420333596e-16, A2.Get(0, 1), t)
	ExpectFloat(-174.99999999999997, A2.Get(1, 1), t)
	ExpectFloat(69.99999999999999, A2.Get(2, 1), t)
	ExpectFloat(-5.335902659890752e-16, A2.Get(0, 2), t)
	ExpectFloat(0.0, A2.Get(1, 2), t)
	ExpectFloat(-35.0, A2.Get(2, 2), t)
}

func TestDecomposeQR(t *testing.T) {
	A := NewArrayMatrix(3, 3)
	A.Set(0, 0, 12)
	A.Set(1, 0, -51)
	A.Set(2, 0, 4)
	A.Set(0, 1, 6)
	A.Set(1, 1, 167)
	A.Set(2, 1, -68)
	A.Set(0, 2, -4)
	A.Set(1, 2, 24)
	A.Set(2, 2, -41)

	Q, R := DecomposeQR(A)

	qIns, qOuts := Q.Shape()
	ExpectInt(3, qIns, t)
	ExpectInt(3, qOuts, t)

	ExpectFloat(-0.8571428571428572, Q.Get(0, 0), t)
	ExpectFloat(0.39428571428571435, Q.Get(1, 0), t)
	ExpectFloat(0.33142857142857146, Q.Get(2, 0), t)
	ExpectFloat(-0.4285714285714286, Q.Get(0, 1), t)
	ExpectFloat(-0.9028571428571428, Q.Get(1, 1), t)
	ExpectFloat(-0.03428571428571425, Q.Get(2, 1), t)
	ExpectFloat(0.28571428571428575, Q.Get(0, 2), t)
	ExpectFloat(-0.17142857142857143, Q.Get(1, 2), t)
	ExpectFloat(0.9428571428571428, Q.Get(2, 2), t)

	rIns, rOuts := R.Shape()
	ExpectInt(3, rIns, t)
	ExpectInt(3, rOuts, t)

	ExpectFloat(-14, R.Get(0, 0), t)
	ExpectFloat(-21, R.Get(1, 0), t)
	ExpectFloat(14, R.Get(2, 0), t)
	ExpectFloat(0, R.Get(0, 1), t)
	ExpectFloat(-175, R.Get(1, 1), t)
	ExpectFloat(70, R.Get(2, 1), t)
	ExpectFloat(0, R.Get(0, 2), t)
	ExpectFloat(0, R.Get(1, 2), t)
	ExpectFloat(-35, R.Get(2, 2), t)

	B := Compose(R, Q) // should be same as A

	aIns, aOuts := A.Shape()
	bIns, bOuts := B.Shape()
	ExpectInt(aIns, bIns, t)
	ExpectInt(aOuts, bOuts, t)
	for o := 0; o < aOuts; o++ {
		for i := 0; i < aIns; i++ {
			ExpectFloat(A.Get(i, o), B.Get(i, o), t)
		}
	}
}

func TestOrdinaryLeastSquares(t *testing.T) {
	X := NewArrayMatrix(2, 2)
	X.Set(0, 0, 1)
	X.Set(1, 0, 0)
	X.Set(0, 1, 1)
	X.Set(1, 1, 2)

	y := NewArrayVector(2)
	y.Set(0, 6)
	y.Set(1, 0)

	theta_hat := OrdinaryLeastSquares(X, y)

	ExpectInt(2, theta_hat.Dimension(), t)
	ExpectFloat(6, theta_hat.Get(0), t)
	ExpectFloat(-3, theta_hat.Get(1), t)
}

func TestOrdinaryLeastSquaresNonSquare(t *testing.T) {
	X := NewArrayMatrix(2, 3)
	X.Set(0, 0, 1)
	X.Set(1, 0, 0)
	X.Set(0, 1, 1)
	X.Set(1, 1, 2)
	X.Set(0, 2, -2)
	X.Set(1, 2, 1)

	y := NewArrayVector(3)
	y.Set(0, 6)
	y.Set(1, 0)
	y.Set(2, -15)

	theta_hat := OrdinaryLeastSquares(X, y)

	ExpectInt(2, theta_hat.Dimension(), t)
	ExpectFloat(6, theta_hat.Get(0), t)
	ExpectFloat(-3, theta_hat.Get(1), t)
}

func BenchmarkFindInputToUpperTriangularInto(b *testing.B) {
	ins := 14
	outs := 506
	x := NewArrayVector(14)
	for i := 0; i < ins; i++ {
		x.Set(i, rand.Float64())
	}

	A := NewArrayMatrix(ins, outs)
	for o := 0; o < outs; o++ {
		for i := 0; i < ins; i++ {
			if i >= o {
				A.Set(i, o, rand.Float64())
			}
		}
	}

	y := ApplyToVector(A, x)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FindInputToUpperTriangularInto(A, y, x)
	}
}
