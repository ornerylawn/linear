// Package linear is where I implement linear algebra for
// deep learning (as in me, the human, learning deeply). Here I take
// the perspective of maps between vector spaces, as opposed
// to the more general tensor network view with higher-rank
// connections.
package linear

import (
	"fmt"
)

// FindInputUpperTriangular finds the input vector that maps to the
// given output vector in the case of an upper triangular map.
func FindInputUpperTriangular(A Matrix, b Matrix) Matrix {
	ins, outs := A.Shape()
	x := NewArrayMatrix(1, ins)
	CheckVector(x)
	CheckUpperTriangular(A)
	CheckVector(b)
	CheckComposable(x, A)
	CheckSameIns(x, b) // redundant since they're both vectors
	CheckSameOuts(A, b)

	if outs < ins {
		panic(fmt.Errorf("less matix outs (%d) than ins (%d)", outs, ins))
	}

	// Since A is upper triangular we can solve the last row on the
	// diagonal (the rest are zeros) by simple division, and then use
	// that to solve the previous row and so on.
	for o := ins - 1; o >= 0; o-- {
		dot := DotProduct(
			Slice(x, 0, 1, o+1, ins),
			Slice(A, o+1, ins, o, o+1))
		numer := b.Get(0, o) - dot
		denom := A.Get(o, o)
		CheckNotCloseToZero(denom)
		x.Set(0, o, numer/denom)
	}

	return x
}

// Householder finds the linear map that takes x to a vector of the
// same length in the direction of e via reflection over their
// bisection.
func Householder(x, e Matrix) Matrix {
	CheckVector(x)
	CheckVector(e)
	CheckSameOuts(x, e)
	_, dim := x.Shape()

	H := Identity(dim)

	xmag := L2Norm(x)
	x0sign := 1.0
	if x.Get(0, 0) < 0.0 {
		x0sign = -1.0
	}

	u := NewArrayMatrix(1, dim)
	for d := 0; d < dim; d++ {
		u.Set(0, d, x.Get(0, d)+x0sign*xmag*e.Get(0, d))
	}
	Normalize(u)

	for o := 0; o < dim; o++ {
		for i := 0; i < dim; i++ {
			H.Set(i, o, H.Get(i, o)-2*u.Get(0, o)*u.Get(0, i))
		}
	}

	return H
}

// DecomposeQR decomposes A into Q*R by transforming it into an upper
// triangular matrix R. Applying the opposite of the transformation,
// which is Q, to R gets you back to A.
func DecomposeQR(A Matrix) (Q Matrix, R Matrix) {
	ins, outs := A.Shape()
	Q = Identity(outs)
	R = Slice(A, 0, ins, 0, outs)
	for i := 0; i < ins; i++ {
		if IsZero(Slice(R, i, i+1, i+1, outs)) {
			continue
		}

		x := Slice(R, i, i+1, i, outs)
		e := BasisVector(outs-i, 0)
		H := Householder(x, e)

		// Extend.
		HE := Identity(outs)
		_, xdim := x.Shape()
		for ho := 0; ho < xdim; ho++ {
			for hi := 0; hi < xdim; hi++ {
				HE.Set(i+hi, i+ho, H.Get(hi, ho))
			}
		}

		R = Apply(HE, R)
		Q = Compose(Dual(HE), Q)
	}
	return Q, R
}

// OrdinaryLeastSquares finds the input (parameters) that when mapped
// (by the dataset inputs) is closest to the output (the dataset
// outputs) in terms of L2 distance.
func OrdinaryLeastSquares(X Matrix, y Matrix) Matrix {
	// X*theta_hat != y, but we want the left to come as close as
	// possible to y, the projection of y onto the column space of X.
	//
	// Dual(X)*y compares y against each of the columns of X,
	// essentially rewriting y in terms of X's columns the best we can.
	//
	// Doing that on the other side too gives us the "normal equation".
	//
	// Dual(X)*X*theta_hat = Dual(X)*y
	//
	// Solving using Inverse(Dual(X) * X) can lead to numerical
	// instability apparently, so instead it is common to use the QR
	// decomposition, using the orthogonality of Q to simplify and the
	// upper-triangularness of R to solve the special case of A*x = b
	// where A is upper triangular by back substitution.
	//
	//     Dual(Q*R)*(Q*R)*theta_hat = Dual(Q*R)*y
	// Dual(R)*Dual(Q)*Q*R*theta_hat = Dual(R)*Dual(Q)*y
	//           Dual(R)*R*theta_hat = Dual(R)*Dual(Q)*y
	//                   R*theta_hat = Dual(Q)*y
	//
	// This is valid only if Dual(R) is invertible so that we can cancel
	// it.
	CheckVector(y)
	Q, R := DecomposeQR(X)
	b := Apply(Dual(Q), y)
	return FindInputUpperTriangular(R, b)
}
