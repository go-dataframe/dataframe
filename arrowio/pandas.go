package arrowio

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/go-dataframe/dataframe"
)

// The "pandas" schema metadata is the JSON document pandas (through
// pyarrow) stores with a table to restore the row labels and the column
// dtypes. This file writes a compatible one and reads the index back from it.

const pandasKey = "pandas"

type pandasMeta struct {
	IndexColumns  []json.RawMessage `json:"index_columns"`
	ColumnIndexes []pandasColumn    `json:"column_indexes"`
	Columns       []pandasColumn    `json:"columns"`
	Creator       map[string]string `json:"creator,omitempty"`
	PandasVersion string            `json:"pandas_version,omitempty"`
}

type pandasColumn struct {
	Name       *string `json:"name"`
	FieldName  *string `json:"field_name"`
	PandasType string  `json:"pandas_type"`
	NumpyType  string  `json:"numpy_type"`
	Metadata   any     `json:"metadata"`
}

// rangeIndex is the descriptor of a range index, which is stored as
// metadata only.
type rangeIndex struct {
	Kind  string  `json:"kind"`
	Name  *string `json:"name"`
	Start int64   `json:"start"`
	Stop  int64   `json:"stop"`
	Step  int64   `json:"step"`
}

// nameOrNull is a JSON name: the empty name is pandas' None.
func nameOrNull(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// pandasType returns the pandas_type, numpy_type and metadata of a column.
// masked reports a validity bitmap, which makes pandas pick its nullable
// dtypes (Int64, boolean) when it reads the column back.
func pandasType(s *dataframe.Series, masked bool, tz string) pandasColumn {
	dt := s.DType()
	name := dt.String()
	c := pandasColumn{PandasType: name, NumpyType: name}
	switch dt {
	case dataframe.Bool:
		if masked {
			c.NumpyType = "boolean"
		}
	case dataframe.Int8, dataframe.Int16, dataframe.Int32, dataframe.Int64:
		if masked {
			c.NumpyType = "I" + name[1:]
		}
	case dataframe.Uint8, dataframe.Uint16, dataframe.Uint32, dataframe.Uint64:
		if masked {
			c.NumpyType = "UI" + name[2:]
		}
	case dataframe.String, dataframe.Period:
		c.PandasType, c.NumpyType = "unicode", "object"
	case dataframe.Datetime:
		c.PandasType, c.NumpyType = "datetime", "datetime64[ns]"
		if tz != "" {
			c.PandasType = "datetimetz"
			c.Metadata = map[string]string{"timezone": tz}
		}
	case dataframe.Category:
		_, cats, _ := s.Categorical()
		c.PandasType, c.NumpyType = "categorical", "int32"
		c.Metadata = map[string]any{"num_categories": len(cats), "ordered": false}
	}
	return c
}

// encodePandas returns the metadata document for the given columns and
// index descriptors (field names, or one range descriptor).
func encodePandas(cols []pandasColumn, index []any) string {
	m := struct {
		IndexColumns  []any             `json:"index_columns"`
		ColumnIndexes []pandasColumn    `json:"column_indexes"`
		Columns       []pandasColumn    `json:"columns"`
		Attributes    map[string]any    `json:"attributes"`
		Creator       map[string]string `json:"creator"`
		PandasVersion string            `json:"pandas_version"`
	}{
		IndexColumns: index,
		ColumnIndexes: []pandasColumn{{
			PandasType: "unicode", NumpyType: "object",
			Metadata: map[string]string{"encoding": "UTF-8"},
		}},
		Columns:       cols,
		Attributes:    map[string]any{},
		Creator:       map[string]string{"library": "go-dataframe/arrowio"},
		PandasVersion: "2.2.0",
	}
	if m.IndexColumns == nil {
		m.IndexColumns = []any{}
	}
	b, _ := json.Marshal(m)
	return string(b)
}

// indexSpec is how the index of a read table is restored: a range, or
// columns turned into levels.
type indexSpec struct {
	rng    *rangeIndex
	fields []string // the columns holding the levels
	names  []string // the level names
}

// decodePandas reads the index part of the "pandas" metadata, if any. A
// table without it gets a default range index (nil spec).
func decodePandas(schema *arrow.Schema) (*indexSpec, error) {
	raw, ok := schema.Metadata().GetValue(pandasKey)
	if !ok {
		return nil, nil
	}
	var m pandasMeta
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrMetadata, err)
	}
	names := map[string]*string{}
	for _, c := range m.Columns {
		if c.FieldName != nil {
			names[*c.FieldName] = c.Name
		}
	}
	spec := &indexSpec{}
	for _, raw := range m.IndexColumns {
		var field string
		if json.Unmarshal(raw, &field) == nil {
			spec.fields = append(spec.fields, field)
			spec.names = append(spec.names, levelName(field, names))
			continue
		}
		var r rangeIndex
		if err := json.Unmarshal(raw, &r); err != nil || r.Kind != "range" {
			return nil, fmt.Errorf("%w: index column %s", ErrMetadata, raw)
		}
		spec.rng = &r
	}
	if spec.rng != nil && spec.fields != nil {
		return nil, fmt.Errorf("%w: a range index among index columns", ErrMetadata)
	}
	return spec, nil
}

// levelName returns the name of the index level stored in a field: the
// metadata name (null is unnamed), else the field name unless it is one of
// the placeholders pandas gives unnamed levels.
func levelName(field string, names map[string]*string) string {
	if n, ok := names[field]; ok {
		if n == nil {
			return ""
		}
		return *n
	}
	if strings.HasPrefix(field, "__index_level_") {
		return ""
	}
	return field
}

// applyIndex sets the index the spec describes on a frame read with a
// range index.
func applyIndex(df *dataframe.DataFrame, spec *indexSpec) (*dataframe.DataFrame, error) {
	n := df.Len()
	if spec.rng != nil {
		r := spec.rng
		name := ""
		if r.Name != nil {
			name = *r.Name
		}
		if r.Start == 0 && r.Step == 1 {
			ix, _ := dataframe.NewRangeIndex(n).Rename(name)
			return df.WithIndex(ix)
		}
		labels := make([]int64, n)
		for i := range labels {
			labels[i] = r.Start + int64(i)*r.Step
		}
		ix, _ := dataframe.NewIndex(name, labels)
		return df.WithIndex(ix)
	}
	if len(spec.fields) == 0 {
		return df, nil
	}
	out, err := df.SetIndex(spec.fields...)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrMetadata, err)
	}
	ix, _ := out.Index().Rename(spec.names...)
	return out.WithIndex(ix)
}
