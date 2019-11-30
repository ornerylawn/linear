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

// Identity makes a new square dimxdim Matrix with ones on the
// diagonal.
func Identity(dim int) Matrix {
	I := NewArrayMatrix(dim, dim)
	for d := 0; d < dim; d++ {
		I.Set(d, d, 1)
	}
	return I
}

// Copy makes a new Matrix with the same shape and entries as the
// given Matrix.
func Copy(A Matrix) Matrix {
	ins, outs := A.Shape()
	B := NewArrayMatrix(ins, outs)
	for o := 0; o < outs; o++ {
		for i := 0; i < ins; i++ {
			B.Set(i, o, A.Get(i, o))
		}
	}
	return B
}

// Dual produces the transpose of A.
func Dual(A Matrix) Matrix {
	// TODO: rather than returning new storage, we could have a wrapper
	// around the existing array that reads backwards. But the user
	// would have to know that it is a reference.
	ins, outs := A.Shape()
	B := NewArrayMatrix(outs, ins)
	for o := 0; o < outs; o++ {
		for i := 0; i < ins; i++ {
			B.Set(o, i, A.Get(i, o))
		}
	}
	return B
}

// Compose returns B*A.
func Compose(A, B Matrix) Matrix {
	aIns, aOuts := A.Shape()
	bIns, bOuts := B.Shape()
	if aOuts != bIns {
		panic(fmt.Errorf("these matrices cannot be composed, %d dimensions vs %d dimensions", aOuts, bIns))
	}
	C := NewArrayMatrix(aIns, bOuts)
	for o := 0; o < bOuts; o++ {
		for i := 0; i < aIns; i++ {
			dot := 0.0
			for k := 0; k < aOuts; k++ {
				dot += A.Get(i, k) * B.Get(k, o)
			}
			C.Set(i, o, dot)
		}
	}
	return C
}

// ApplyToMatrix returns A*X.
func ApplyToMatrix(A, X Matrix) Matrix {
	return Compose(X, A)
}

// ApplyToVector returns A*x.
func ApplyToVector(A Matrix, x Vector) Vector {
	ins, outs := A.Shape()
	if x.Dimension() != ins {
		panic(fmt.Errorf("this matrix with %d inputs cannot be applied vector with %d dimensions", ins, outs))
	}
	v := NewArrayVector(outs)
	for o := 0; o < outs; o++ {
		dot := 0.0
		for i := 0; i < ins; i++ {
			dot += A.Get(i, o) * x.Get(i)
		}
		v.Set(o, dot)
	}
	return v
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

// Normalize divides out the L2Norm, yielding a vector in the same
// direction with unit length.
func Normalize(v Vector) Vector {
	mag := L2Norm(v)
	dims := v.Dimension()
	u := NewArrayVector(dims)
	for d := 0; d < dims; d++ {
		u.Set(d, v.Get(d)/mag)
	}
	return u
}

// SolveUpperTriangular solves Ax = b when A is upper triangular.
func SolveUpperTriangular(A Matrix, b Vector) Vector {
	ins, outs := A.Shape()
	if outs < ins {
		panic(fmt.Errorf("can't solve upper triangular with less outs (%d) than ins (%d)", outs, ins))
	}
	if b.Dimension() != outs {
		panic(fmt.Errorf("expected b to have same dims (%d) as A has outs (%d)", b.Dimension(), outs))
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
	x := NewArrayVector(ins)
	for o := ins - 1; o >= 0; o-- {
		dot := 0.0
		for i := o + 1; i < ins; i++ {
			dot += x.Get(i) * A.Get(i, o)
		}
		numer := b.Get(o) - dot
		denom := A.Get(o, o)
		if denom < 1e-9 {
			panic(fmt.Errorf("entry is too small for stability (%d, %d is %f)", o, o, denom))
		}
		x.Set(o, numer/denom)
	}
	return x
}

// Householder returns the householder reflection operator for the
// [d:, d:] submatrix of A (extended back to the shape of A) which
// when applied to A zeros subdiagonal elements of column d.
func Householder(A Matrix, d int) Matrix {
	ins, outs := A.Shape()
	if d >= ins-1 {
		panic(fmt.Errorf("there is no householder reflection for this submatrix (%d, %d submatrix %d)", ins, outs, d))
	}

	// x = A[d, d:] which we want to set to [* 0 ... 0]
	// e = [1 0 ... 0]
	// u = x - mag(x)*e which is just [x0-magx x1 ... xk]
	// v = normalize(u)
	// Q = I - 2 * v * dual(v).

	xdim := outs - d
	x := NewArrayVector(xdim)
	for k := 0; k < xdim; k++ {
		x.Set(k, A.Get(d, d+k))
	}
	xmag := L2Norm(x)
	x0sign := 1.0
	if x.Get(0) < 0.0 {
		x0sign = -1.0
	}

	u := NewArrayVector(xdim)
	u.Set(0, x.Get(0)+x0sign*xmag)
	for k := 1; k < xdim; k++ {
		u.Set(k, x.Get(k))
	}

	v := Normalize(u)
	Q := Identity(outs)

	for k := 0; k < xdim; k++ {
		o := d + k
		for j := 0; j < xdim; j++ {
			i := d + j
			Q.Set(i, o, Q.Get(i, o)-2*v.Get(k)*v.Get(j))
		}
	}

	return Q
}

// DecomposeQR decomposes A into Q*R by transforming it into an upper
// triangular matrix R. Applying the opposite of the transformation,
// which is Q, to R gets you back to A. If you are trying to solve Ax
// = b, this gives QRx = b, which yields Rx = Dual(Q)*b because Q is
// orthogonal. Since R is upper triangular, this last equation is
// solvable by SolveUpperTriangular.
func DecomposeQR(A Matrix) (Q, R Matrix) {
	ins, outs := A.Shape()
	Q = Identity(outs)
	R = Copy(A)

	// We transform each column of A to have zeros below the diagonal
	// using householder reflections. The resulting upper triangular
	// matrix is R, and the composition of the transpose of each
	// householder reflection is Q.

	for i := 0; i < ins; i++ {
		// If subdiagonal entries are zero we can skip the reflection.
		zero := true
		for o := i + 1; o < outs; o++ {
			if R.Get(i, o) != 0.0 {
				zero = false
				break
			}
		}
		if zero {
			continue
		}

		H := Householder(R, i)
		R = ApplyToMatrix(H, R)
		Q = Compose(Dual(H), Q)
	}
	return Q, R
}
