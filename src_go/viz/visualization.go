/*
Visualization of outputs:
- overlays optical flow vectors over original images
- generates gifs for a visualization of optical flow movement

References:
- Go Image Library: https://pkg.go.dev/image
- Go GIF Library from Image: https://pkg.go.dev/image/gif
*/
package viz

import (
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"image/gif"
	"image/png"
	"os"

	"gocv.io/x/gocv"
)

// SaveImage saves an image to a file in PNG format
func SaveImage(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

// LoadImage loads an image from a PNG file
func LoadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return png.Decode(f)
}

func DrawOpticalFlow(img *gocv.Mat, prevPts, nextPts []image.Point) {
	/*
		Overlay green optical flow vectors on sequential frames to show movement
	*/

	// Convert grayscale to RGB if needed
	if img.Channels() == 1 {
		gocv.CvtColor(*img, img, gocv.ColorGrayToBGR)
	}

	// Draw green flow vectors on the image
	// Arrow is drawn from the previous points to the next points to show movement
	green := color.RGBA{G: 255, A: 255}
	for i := range prevPts {
		pt1 := prevPts[i]
		pt2 := nextPts[i]
		gocv.ArrowedLine(img, pt1, pt2, green, 1)
	}
}

// MatToImage converts a gocv.Mat to image.Image
func MatToImage(mat gocv.Mat) image.Image {
	// Ensure the image is in BGR format
	if mat.Channels() == 1 {
		rgb := gocv.NewMat()
		defer rgb.Close()
		gocv.CvtColor(mat, &rgb, gocv.ColorGrayToBGR)
		mat = rgb
	}

	// Convert to image.RGBA
	bounds := image.Rect(0, 0, mat.Cols(), mat.Rows())
	img := image.NewRGBA(bounds)
	for y := 0; y < mat.Rows(); y++ {
		for x := 0; x < mat.Cols(); x++ {
			vec := mat.GetVecbAt(y, x)
			img.Set(x, y, color.RGBA{
				B: vec[0], // OpenCV uses BGR format
				G: vec[1],
				R: vec[2],
				A: 255,
			})
		}
	}
	return img
}

// SaveAsGIF saves a sequence of frames as an animated GIF
func SaveAsGIF(frames []image.Image, outputPath string, delay int) error {
	/*
		Saves a sequence of frames as an animated GIF
		From GIF in Image library: https://pkg.go.dev/image/gif
	*/
	if len(frames) == 0 {
		return nil
	}

	f, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer f.Close()

	anim := gif.GIF{}

	for _, frame := range frames {
		bounds := frame.Bounds()
		palettedImg := image.NewPaletted(bounds, palette.Plan9)
		draw.Draw(palettedImg, bounds, frame, bounds.Min, draw.Over)
		anim.Image = append(anim.Image, palettedImg)
		anim.Delay = append(anim.Delay, delay/10)
	}

	return gif.EncodeAll(f, &anim)
}
