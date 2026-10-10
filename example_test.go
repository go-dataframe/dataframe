package dataframe_test

import (
	"fmt"

	"github.com/go-dataframe/dataframe"
)

func Example() {
	df := dataframe.Must(dataframe.New(
		dataframe.Col("city", []string{"Lyon", "Paris", "Nice", "Paris", "Lyon"}),
		dataframe.Col("year", []int{2023, 2023, 2024, 2024, 2024}),
		dataframe.Col("rain", []float64{830.5, 640.2, 760.1, 598.7, 810.0}),
	))
	fmt.Println(df.Head(3))

	// df.groupby("city").mean(numeric_only=True)
	g := dataframe.Must(df.GroupBy("city"))
	fmt.Println(dataframe.Must(g.Mean()).ResetIndex(false))

	// df.query("rain > 700 and year == 2024")
	fmt.Println(dataframe.Must(df.Query("rain > 700 and year == 2024")))
	// Output:
	//     city  year   rain
	// 0   Lyon  2023  830.5
	// 1  Paris  2023  640.2
	// 2   Nice  2024  760.1
	//     city    year    rain
	// 0   Lyon  2023.5  820.25
	// 1   Nice  2024.0  760.10
	// 2  Paris  2023.5  619.45
	//    city  year   rain
	// 2  Nice  2024  760.1
	// 4  Lyon  2024  810.0
}

func ExampleMerge() {
	left := dataframe.Must(dataframe.New(
		dataframe.Col("id", []int{1, 2, 3}),
		dataframe.Col("name", []string{"ada", "bob", "cy"}),
	))
	right := dataframe.Must(dataframe.New(
		dataframe.Col("id", []int{3, 1, 1}),
		dataframe.Col("score", []float64{7.5, 9, 8}),
	))
	// pd.merge(left, right, on="id", how="left")
	fmt.Println(dataframe.Must(dataframe.Merge(left, right, dataframe.MergeOptions{On: []string{"id"}, How: "left"})))
	// Output:
	//    id name  score
	// 0   1  ada    9.0
	// 1   1  ada    8.0
	// 2   2  bob    NaN
	// 3   3   cy    7.5
}

func ExampleDataFrame_Resample() {
	ts := dataframe.Must(dataframe.ToDatetime(dataframe.Must(dataframe.NewSeries("when",
		[]string{"2024-01-01 09:00", "2024-01-01 17:30", "2024-01-03 08:00"}))))
	df := dataframe.Must(dataframe.New(dataframe.Col("when", ts), dataframe.Col("kwh", []float64{1.5, 2, 4})))
	df = dataframe.Must(df.SetIndex("when"))
	// df.resample("D").sum()
	fmt.Println(dataframe.Must(dataframe.Must(df.Resample("D")).Sum()).ResetIndex(false))
	// Output:
	//         when  kwh
	// 0 2024-01-01  3.5
	// 1 2024-01-02  0.0
	// 2 2024-01-03  4.0
}
