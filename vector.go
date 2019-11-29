package linear

type Vector interface {
	Dimensions() int
	Get(i int) float64
	Set(i int, value float64)
}

type ArrayVector struct {
	array []float64
	dims  int
}

func NewArrayVector(dims int) Vector {
	return &ArrayVector{
		array: make([]float64, dims),
		dims:  dims,
	}
}

func (v *ArrayVector) Dimensions() int          { return v.dims }
func (v *ArrayVector) Get(i int) float64        { return v.array[i] }
func (v *ArrayVector) Set(i int, value float64) { v.array[i] = value }
