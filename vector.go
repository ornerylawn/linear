package linear

type Vector interface {
	Dimensions() int
	Get(i int) float64
	Set(i int, value float64)
}

type ArrayVector []float64

func NewArrayVector(dims int) Vector {
	return ArrayVector(make([]float64, dims))
}

func (v ArrayVector) Dimensions() int          { return len(v) }
func (v ArrayVector) Get(i int) float64        { return v[i] }
func (v ArrayVector) Set(i int, value float64) { v[i] = value }
