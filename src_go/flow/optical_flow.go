/*
Optical Flow estimation for medical imaging (for our DICOM files) using Lucas-Kanade method.
This implementation mimics functionality provided by pythons OpenCV library.

Based on Optical Flow in OpenCV:
https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
  - calcOpticalFlowPyrLK(): https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
  - go implementation: https://github.com/hybridgroup/gocv/blob/release/video.go
    Optical Flow Resourses:

- A Comprehensive guide to Motion Estimation with Optical Flow: https://nanonets.com/blog/optical-flow/
*/
package flow

import (
	"image"

	"gocv.io/x/gocv" // Library for using OpenCV in Go
)

// Defines parameters to be used in the optical flow algorithm
type OpticalFlowParams struct {
	WinSize       image.Point // size of search window
	MaxLevel      int         // max pyramid level
	MaxIterations int         // termination criteria: defines maximum iterations of alg.
	Epsilon       float64     // minimum optical flow movement (between original and moved point)
}

// Define parameters based on our usage: DICOM files in medical imaging
func DefaultOpticalFlowParams() OpticalFlowParams {
	return OpticalFlowParams{
		WinSize:       image.Point{X: 21, Y: 21},
		MaxLevel:      3,
		MaxIterations: 30,
		Epsilon:       0.3,
	}
}

// ComputeOpticalFlow calculates the optical flow between two consecutive frames
// Based on calcOpticalFlowPyrLK() function in OpenCV, which uses iteravtive Lucas-Kanade method with pyramids
// https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
func ComputeOpticalFlow(prevImg, nextImg gocv.Mat, prevPts []image.Point, params OpticalFlowParams) ([]image.Point, []bool) {
	if len(prevPts) == 0 {
		return []image.Point{}, []bool{}
	}

	// Create matrices for points
	prevPointsMat := gocv.NewMatWithSize(len(prevPts), 1, gocv.MatTypeCV32FC2)
	nextPointsMat := gocv.NewMatWithSize(len(prevPts), 1, gocv.MatTypeCV32FC2)
	statusMat := gocv.NewMat()
	errMat := gocv.NewMat()

	defer prevPointsMat.Close()
	defer nextPointsMat.Close()
	defer statusMat.Close()
	defer errMat.Close()

	// Convert points to Mat format
	for i, pt := range prevPts {
		prevPointsMat.SetFloatAt(i, 0, float32(pt.X))
		prevPointsMat.SetFloatAt(i, 1, float32(pt.Y))
	}

	// Calculate optical flow
	// Using OpenCV in go: https://github.com/hybridgroup/gocv/blob/release/video.go
	gocv.CalcOpticalFlowPyrLK(
		prevImg,
		nextImg,
		prevPointsMat,
		nextPointsMat,
		&statusMat,
		&errMat,
	)

	// Convert output to Go native types
	trackedPoints := make([]image.Point, len(prevPts))
	tracked := make([]bool, len(prevPts))

	for i := 0; i < len(prevPts); i++ {
		nextX := nextPointsMat.GetFloatAt(i, 0)
		nextY := nextPointsMat.GetFloatAt(i, 1)
		status := statusMat.GetUCharAt(i, 0)

		trackedPoints[i] = image.Point{
			X: int(nextX),
			Y: int(nextY),
		}

		// Existance and Threshold check
		prevX := prevPointsMat.GetFloatAt(i, 0)
		prevY := prevPointsMat.GetFloatAt(i, 1)
		tracked[i] = status != 0 && // Check if point was tracked
			abs(nextX-prevX) < float32(params.Epsilon) && // Check if movement is within epsilon
			abs(nextY-prevY) < float32(params.Epsilon)
	}

	return trackedPoints, tracked
}

// little helper function
func abs(x float32) float32 {
	/* Gets absolute value of a float32 */
	if x < 0 {
		return -x
	}
	return x
}
