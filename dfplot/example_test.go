package dfplot_test

import (
	"bytes"
	"fmt"

	"github.com/go-dataframe/dataframe"
	"github.com/go-dataframe/dataframe/dfplot"
)

func Example() {
	df := dataframe.Must(dataframe.New(
		dataframe.Col("day", []int64{1, 2, 3, 4}),
		dataframe.Col("sales", []float64{10, 12, 9, 15}),
		dataframe.Col("costs", []float64{8, 9, 9, 10}),
	))
	df, _ = df.SetIndex("day")

	fig, err := dfplot.Line(df, "sales", "costs")
	if err != nil {
		fmt.Println(err)
		return
	}
	var png bytes.Buffer
	if err := fig.SavePNG(&png); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("axes:", len(fig.Axes()), "png:", png.Len() > 0)

	// Other kinds: dfplot.Bar, dfplot.Barh, dfplot.Hist, dfplot.Area,
	// dfplot.Scatter and the Series variants.
	_, err = dfplot.Scatter(df, "sales", "costs", dfplot.WithTitle("sales vs costs"))
	fmt.Println("scatter error:", err)
	// Output:
	// axes: 1 png: true
	// scatter error: <nil>
}
