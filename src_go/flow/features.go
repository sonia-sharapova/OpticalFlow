/*
Feature Detection: Using Shi-Tomasi Corner Detection
- feature detection based on thresholding the minimum eigenvalue from pixel intensities
- OpenCV Theory and breakdown: https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html

Implementation:
Based on OpenCV's GoodFeaturesToTrack function, which detects prominent corners in an image using Shi-Tomasi.
- From gocv Library: https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
*/

package flow

import (
	"image" // 2D image library

	"gocv.io/x/gocv" // OpenCV for Go
)

// Define parameters for Shi-Tomasi corner detection.
type ShiTomasiParams struct {
	MaxCorners   int
	QualityLevel float64
	MinDistance  float64
}

// DetectFeatures detects Shi-Tomasi corners in the given image (as 'Mat', an n-dimensional dense numerical array)
func DetectFeatures(img gocv.Mat, params ShiTomasiParams) ([]image.Point, error) {
	// Create output matrix
	corners := gocv.NewMat() // Create a new empty Mat for detected corners
	defer corners.Close()

	// Perform corner detection using GoodFeaturesToTrack
	gocv.GoodFeaturesToTrack(
		img,                 // input image(8-bit)
		&corners,            // the deected corners
		params.MaxCorners,   // maximum number of corners to return
		params.QualityLevel, // minimum accepted quality level
		params.MinDistance,  // minimum distance between detected corners
	)

	// Convert detected features into image.Point array (X, Y coordinate pair)
	points := make([]image.Point, 0)
	for i := 0; i < corners.Rows(); i++ {
		x := int(corners.GetFloatAt(i, 0))
		y := int(corners.GetFloatAt(i, 1))
		points = append(points, image.Point{X: x, Y: y})
	}

	return points, nil
}

// Commonly used parameters for feature detection
func DefaultShiTomasiParams() ShiTomasiParams {
	return ShiTomasiParams{
		MaxCorners:   150,
		QualityLevel: 0.03,
		MinDistance:  7,
	}
}
