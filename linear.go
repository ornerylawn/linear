// Package linear is where I implement linear algebra operations for
// deep learning (as in me, the human, learning deeply).
package linear

import (
	"fmt"
	"math"
)

// TODO: each time we want a new matrix we assume that the user wants
// an arrayMatrix, but we should really have the user determine the
// implementation.

// FindInputToUpperTriangularInto writes the input into x that when
// mapped by upper triangular A yields b. In other words, it solves
// A*x = b for the special case that A is upper triangular.
func FindInputToUpperTriangularInto(A Matrix, b, x Vector) {
	ins, outs := A.Shape()
	if outs < ins {
		panic(fmt.Errorf("can't solve upper triangular with less outs (%d) than ins (%d)", outs, ins))
	}
	if b.Dimension() != outs {
		panic(fmt.Errorf("expected b to have same dims (%d) as A has outs (%d)", b.Dimension(), outs))
	}
	if x.Dimension() != ins {
		panic(fmt.Errorf("dimension mismatch %d vs %d", x.Dimension(), ins))
	}
	for i := 0; i < ins; i++ {
		for o := i + 1; o < outs; o++ {
			if A.Get(i, o) != 0.0 {
				panic(fmt.Errorf("expected upper triangular but found nonzero (%d, %d is %f)", i, o, A.Get(i, o)))
			}
		}
	}
	// Since A is upper triangular we can solve the last row on the
	// diagonal (the rest are zeros) by simple division, and then use
	// that to solve the previous row and so on.
	for o := ins - 1; o >= 0; o-- {
		dot := 0.0
		for i := o + 1; i < ins; i++ {
			dot += x.Get(i) * A.Get(i, o)
		}
		numer := b.Get(o) - dot
		denom := A.Get(o, o)
		if math.Abs(denom) < 1e-9 {
			panic(fmt.Errorf("entry is too small for stability (%d, %d is %f)", o, o, denom))
		}
		x.Set(o, numer/denom)
	}
}

// FindInputToUpperTriangular finds the input x that when mapped by
// upper triangular A yields b. In other words, it solves A*x = b for
// the special case that A is upper triangular.
func FindInputToUpperTriangular(A Matrix, b Vector) Vector {
	ins, _ := A.Shape()
	x := NewArrayVector(ins)
	FindInputToUpperTriangularInto(A, b, x)
	return x
}

// HouseholderInto.
func HouseholderInto(x, e Vector, H Matrix) {
	dim := x.Dimension()
	if e.Dimension() != dim {
		panic(fmt.Errorf("dimension mismatch %d vs %d", x.Dimension(), e.Dimension()))
	}
	hins, houts := H.Shape()
	if hins != houts {
		panic(fmt.Errorf("dimension mismatch %d vs %d", hins, houts))
	}
	if hins != dim {
		panic(fmt.Errorf("dimension mismatch %d vs %d", hins, dim))
	}

	xmag := L2Norm(x)
	x0sign := 1.0
	if x.Get(0) < 0.0 {
		x0sign = -1.0
	}

	// TODO: can we use H's memory rather than creating new vectors
	// here?

	u := NewArrayVector(dim)
	for d := 0; d < dim; d++ {
		u.Set(d, x.Get(d)+x0sign*xmag*e.Get(d))
	}

	v := Normalize(u)

	IdentityInto(H)
	for o := 0; o < dim; o++ {
		for i := 0; i < dim; i++ {
			H.Set(i, o, H.Get(i, o)-2*v.Get(o)*v.Get(i))
		}
	}
}

// Householder produces the matrix of the linear map that reflects the
// given vector x through the hyperplane that bisects x and the given
// vector e.
func Householder(x, e Vector) Matrix {
	H := Identity(x.Dimension())
	HouseholderInto(x, e, H)
	return H
}

// DecomposeIntoQR decomposes A into Q*R by transforming it into an
// upper triangular matrix R. Applying the opposite of the
// transformation, which is Q, to R gets you back to A.
func DecomposeIntoQR(A, Q, R Matrix) {
	// TODO: use less temporary memory.
	ins, outs := A.Shape()
	qIns, qOuts := Q.Shape()
	if qIns != outs || qOuts != outs {
		panic(fmt.Errorf("expected (%d, %d) but got (%d, %d)", outs, outs, qIns, qOuts))
	}
	rIns, rOuts := R.Shape()
	if rIns != ins || rOuts != outs {
		panic(fmt.Errorf("expected (%d, %d) but got (%d, %d)", ins, outs, rIns, rOuts))
	}

	tmpQ := Identity(outs)
	tmpR := Slice(A, 0, ins, 0, outs)

	// We transform each column of A to have zeros below the diagonal
	// using a householder reflection. The resulting upper triangular
	// matrix is R, and the composition of the transpose of each
	// householder reflection is Q.

	for i := 0; i < ins; i++ {
		// If subdiagonal entries in the column are zero we don't need a
		// reflection to make them zero.
		sub := Slice(tmpR, i, i+1, i+1, outs)
		if IsZero(sub) {
			continue
		}

		x := VectorFromColumn(Slice(tmpR, i, i+1, i, outs))
		e := BasisVector(x.Dimension(), 0)
		H := Identity(outs)
		HouseholderInto(x, e, Slice(H, i, outs, i, outs))
		tmpR = ApplyToMatrix(H, tmpR)
		tmpQ = Compose(Dual(H), tmpQ)
	}

	CopyInto(tmpQ, Q)
	CopyInto(tmpR, R)
}

// DecomposeQR decomposes A into Q*R by transforming it into an upper
// triangular matrix R. Applying the opposite of the transformation,
// which is Q, to R gets you back to A.
func DecomposeQR(A Matrix) (Q, R Matrix) {
	ins, outs := A.Shape()
	Q = NewArrayMatrix(outs, outs)
	R = NewArrayMatrix(ins, outs)
	DecomposeIntoQR(A, Q, R)
	return Q, R
}

// OrdinaryLeastSquares finds the input vector that when mapped by X
// yields the minimum sum of squared differences from y.
func OrdinaryLeastSquares(X Matrix, y Vector) Vector {
	// The argmin is given by the "normal equation".
	//
	// Dual(X) * X * theta_hat = Dual(X) * y
	//
	// Solving using Inverse(Dual(X) * X) can lead to numerical
	// instability, so instead it is common to use the QR decomposition,
	// using the orthogonality of Q to simplify and the
	// upper-triangularness of R to solve the special case of A*x = b
	// where A is upper triangular.
	//
	//     Dual(Q * R) * (Q * R) * theta_hat = Dual(Q * R) * y
	// Dual(R) * Dual(Q) * Q * R * theta_hat = Dual(R) * Dual(Q) * y
	//                         R * theta_hat = Dual(Q) * y
	Q, R := DecomposeQR(X)
	b := ApplyToVector(Dual(Q), y)
	return FindInputToUpperTriangular(R, b)
}

// LinearGaussianMaximumLikelihoodEstimate finds the argmax of the
// likelihood function for a data set generated by a linear process
// with Gaussian noise, returning the linear parameters of the
// selected process.
func LinearGaussianMaximumLikelihoodEstimate(X Matrix, y Vector) Vector {
	// The argmax of the likelihood function turns out to be the same as
	// the argmin of the squared error function because of the squared
	// difference term in the Gaussian distribution.
	return OrdinaryLeastSquares(X, y)
}

// LinearRegression models a process that yields a single value from
// many inputs, by selecting from the family of linear models. Given a
// vectorized dataset, where each row is a data point, it finds the
// linear combination of X's columns (input features) that best
// matches y (known outputs). Applying the learned linear combination
// to a new input point provides a good estimate of the real process'
// unknown output value if the real process is indeed approximately
// linear.
func LinearRegression(X Matrix, y Vector) Vector {
	// Assume that the data-generating process is actually linear but
	// with Gaussian error.
	return LinearGaussianMaximumLikelihoodEstimate(X, y)
}
