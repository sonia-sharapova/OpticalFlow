/*
Package to load and preprocess DICOM images (files with '.dcm' extension) for further analysis.
I first wrote this in python (supplies in repository). This package matches python's functionality but in Go.

Uses DICOM processing from the 'github.com/suyashkumar/dicom' to read DICOM files and extract pixel data.

My Data:

	DICOM format from AMRGAtlas
	- 2D integer array structure

Functions:
- LoadDICOMFiles: finds all DICOM files in a given directory
- PreprocessDICOM: reads a DICOM file and returns a preprocessed OpenCV matrix

Helper Function:
- getIntValue: extracts integer value from a DICOM element

  - Used Generative AI to assist with handling my specific data
*/

package dicom

import (
	"bytes"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"

	"github.com/suyashkumar/dicom"
	"github.com/suyashkumar/dicom/pkg/frame"
	"github.com/suyashkumar/dicom/pkg/tag"
	"gocv.io/x/gocv"
)

// Use a sync.Pool for Mat objects to reduce allocation overhead
var matPool = sync.Pool{
	New: func() interface{} {
		return gocv.NewMat()
	},
}

type FrameData struct {
	Mat      gocv.Mat
	Features []image.Point
}

// LoadDICOMFiles finds all DICOM files in the given directory
func LoadDICOMFiles(directory string) ([]string, error) {
	var files []string
	err := filepath.Walk(directory, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if strings.HasSuffix(strings.ToLower(info.Name()), ".dcm") && !strings.HasPrefix(info.Name(), "._") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return files, nil
}

// getIntValue safely extracts an integer value from a DICOM element
func getIntValue(value interface{}) (int, error) {
	switch v := value.(type) {
	case uint16:
		return int(v), nil
	case int:
		return v, nil
	case []int:
		if len(v) > 0 {
			return v[0], nil
		}
		return 0, fmt.Errorf("empty integer array")
	case []uint16:
		if len(v) > 0 {
			return int(v[0]), nil
		}
		return 0, fmt.Errorf("empty uint16 array")
	default:
		return 0, fmt.Errorf("unsupported value type for dimension: %T", value)
	}
}

// PreprocessDICOM reads a DICOM file and returns a preprocessed OpenCV matrix
func PreprocessDICOM(filePath string, fixedSize image.Point) (gocv.Mat, error) {
	// Read the file into memory first
	data, err := os.ReadFile(filePath)
	if err != nil {
		return gocv.Mat{}, fmt.Errorf("failed to read DICOM file: %w", err)
	}

	return ProcessPreloadedDICOM(data, fixedSize)
}

// ProcessPreloadedDICOM processes DICOM data that's already in memory
// ProcessPreloadedDICOM processes DICOM data that's already in memory
// ProcessPreloadedDICOM processes DICOM data that's already in memory
func ProcessPreloadedDICOM(data []byte, fixedSize image.Point) (gocv.Mat, error) {
	dataSize := int64(len(data))
	frameChannel := make(chan *frame.Frame, 1)

	dataset, err := dicom.Parse(bytes.NewReader(data), dataSize, frameChannel)
	if err != nil {
		return gocv.Mat{}, fmt.Errorf("failed to parse DICOM data: %w", err)
	}

	// Get dimensions
	rows, err := dataset.FindElementByTag(tag.Rows)
	if err != nil {
		return gocv.Mat{}, fmt.Errorf("failed to find rows: %w", err)
	}
	height, err := getIntValue(rows.Value.GetValue())
	if err != nil {
		return gocv.Mat{}, fmt.Errorf("failed to get height: %w", err)
	}

	cols, err := dataset.FindElementByTag(tag.Columns)
	if err != nil {
		return gocv.Mat{}, fmt.Errorf("failed to find columns: %w", err)
	}
	width, err := getIntValue(cols.Value.GetValue())
	if err != nil {
		return gocv.Mat{}, fmt.Errorf("failed to get width: %w", err)
	}

	// Get pixel data
	pixelDataElement, err := dataset.FindElementByTag(tag.PixelData)
	if err != nil {
		return gocv.Mat{}, fmt.Errorf("failed to find pixel data: %w", err)
	}

	var pixels []uint16
	pixelData := pixelDataElement.Value.GetValue()

	switch data := pixelData.(type) {
	case dicom.PixelDataInfo:
		if len(data.Frames) == 0 {
			return gocv.Mat{}, fmt.Errorf("no frames in pixel data")
		}

		frame := data.Frames[0]
		frameVal := reflect.ValueOf(frame).Elem()

		nativeData := frameVal.FieldByName("NativeData")
		if nativeData.IsValid() {
			dataField := nativeData.FieldByName("Data")
			if dataField.IsValid() {
				rawData := dataField.Interface()

				switch v := rawData.(type) {
				case [][]int:
					// Handle flattened array
					if len(v) == height*width && len(v[0]) == 1 {
						pixels = make([]uint16, height*width)
						for i := 0; i < height*width; i++ {
							if v[i][0] < 0 {
								pixels[i] = 0
							} else if v[i][0] > 65535 {
								pixels[i] = 65535
							} else {
								pixels[i] = uint16(v[i][0])
							}
						}
					} else {
						return gocv.Mat{}, fmt.Errorf("unexpected dimensions: %dx%d", len(v), len(v[0]))
					}
				default:
					return gocv.Mat{}, fmt.Errorf("unsupported pixel data type: %T", v)
				}
			}
		}
	}

	if pixels == nil {
		return gocv.Mat{}, fmt.Errorf("failed to extract pixel data")
	}

	// Create Mat and copy data
	mat := gocv.NewMatWithSize(height, width, gocv.MatTypeCV16UC1)
	if mat.Empty() {
		return gocv.Mat{}, fmt.Errorf("failed to create Mat")
	}

	// Convert pixels to bytes
	frameBytes := make([]byte, len(pixels)*2)
	for i, p := range pixels {
		frameBytes[i*2] = byte(p)
		frameBytes[i*2+1] = byte(p >> 8)
	}

	// Create Mat from bytes
	newMat, err := gocv.NewMatFromBytes(height, width, gocv.MatTypeCV16UC1, frameBytes)
	if err != nil {
		mat.Close()
		return gocv.Mat{}, fmt.Errorf("failed to create Mat from bytes: %w", err)
	}

	// Convert to 8-bit
	converted := gocv.NewMat()
	newMat.ConvertTo(&converted, gocv.MatTypeCV8UC1)
	newMat.Close()

	// Normalize
	normalized := gocv.NewMat()
	gocv.Normalize(converted, &normalized, 0, 255, gocv.NormMinMax)
	converted.Close()

	// Equalize histogram
	equalized := gocv.NewMat()
	gocv.EqualizeHist(normalized, &equalized)
	normalized.Close()

	// Resize
	resized := gocv.NewMat()
	gocv.Resize(equalized, &resized, fixedSize, 0, 0, gocv.InterpolationLinear)
	equalized.Close()

	return resized, nil
}
