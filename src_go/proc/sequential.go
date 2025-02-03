package proc

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"time"

	"proj3/dicom"
	"proj3/flow"
	"proj3/viz"

	"gocv.io/x/gocv"
)

// ProcessTiming stores timing information for different stages
type ProcessTiming struct {
	FileIO        time.Duration
	Preprocessing time.Duration
	FeatureDetect time.Duration
	OpticalFlow   time.Duration
	Visualization time.Duration
}

// SequentialProcessor implements sequential processing
type SequentialProcessor struct {
	BaseProcessor
}

// NewSequentialProcessor creates a new sequential processor
func NewSequentialProcessor() *SequentialProcessor {
	return &SequentialProcessor{
		BaseProcessor: NewBaseProcessor(),
	}
}

// ProcessDataset implements the sequential version
func (p *SequentialProcessor) ProcessDataset(inputDir, outputDir string) error {
	// Process each subfolder
	entries, err := filepath.Glob(filepath.Join(inputDir, "*"))
	if err != nil {
		return fmt.Errorf("failed to read directory: %w", err)
	}

	for _, entry := range entries {
		if info, err := os.Stat(entry); err != nil || !info.IsDir() {
			continue
		}

		if err := p.processFolder(entry, outputDir); err != nil {
			fmt.Printf("Error processing folder %s: %v\n", filepath.Base(entry), err)
			continue // Continue with next folder even if one fails
		}
	}

	return nil
}

func (p *SequentialProcessor) processFolder(inputFolder, outputDir string) error {
	var timing ProcessTiming
	start := time.Now()

	// Create output folder
	folderName := filepath.Base(inputFolder)
	outputFolder := filepath.Join(outputDir, folderName)
	if err := os.MkdirAll(outputFolder, 0755); err != nil {
		return fmt.Errorf("failed to create output folder: %w", err)
	}

	// Load DICOM files
	ioStart := time.Now()
	files, err := dicom.LoadDICOMFiles(inputFolder)
	if err != nil {
		return fmt.Errorf("failed to load DICOM files: %w", err)
	}
	timing.FileIO = time.Since(ioStart)

	if len(files) < 2 {
		return fmt.Errorf("not enough frames to compute optical flow")
	}

	var frames []image.Image
	var prevMat, currentMat gocv.Mat
	var prevFeatures []image.Point

	// Process first frame
	t := time.Now()
	prevMat, err = dicom.PreprocessDICOM(files[0], image.Point{X: 256, Y: 256})
	timing.Preprocessing += time.Since(t)

	if err != nil {
		return fmt.Errorf("failed to process first frame: %w", err)
	}
	defer prevMat.Close()

	t = time.Now()
	prevFeatures, err = flow.DetectFeatures(prevMat, p.featureParams)
	timing.FeatureDetect += time.Since(t)

	if err != nil {
		return fmt.Errorf("failed to detect features in first frame: %w", err)
	}

	// Process subsequent frames
	for i := 1; i < len(files); i++ {
		t = time.Now()
		currentMat, err = dicom.PreprocessDICOM(files[i], image.Point{X: 256, Y: 256})
		timing.Preprocessing += time.Since(t)

		if err != nil {
			continue
		}

		// Compute optical flow
		t = time.Now()
		nextPoints, status := flow.ComputeOpticalFlow(prevMat, currentMat, prevFeatures, p.flowParams)
		timing.OpticalFlow += time.Since(t)

		// Create visualization frame
		t = time.Now()
		var goodPrev, goodNext []image.Point
		for j, tracked := range status {
			if tracked {
				goodPrev = append(goodPrev, prevFeatures[j])
				goodNext = append(goodNext, nextPoints[j])
			}
		}

		visualMat := currentMat.Clone()
		viz.DrawOpticalFlow(&visualMat, goodPrev, goodNext)
		frame := viz.MatToImage(visualMat)
		frames = append(frames, frame)
		visualMat.Close()
		timing.Visualization += time.Since(t)

		// Update for next iteration
		prevMat.Close()
		prevMat = currentMat
		prevFeatures = nextPoints
	}

	// Save results
	if len(frames) > 0 {
		outputPath := filepath.Join(outputFolder, folderName+".gif")
		if err := viz.SaveAsGIF(frames, outputPath, 500); err != nil {
			return fmt.Errorf("failed to save GIF: %w", err)
		}
	}

	totalTime := time.Since(start)
	fmt.Printf("\nTiming for folder %s:\n", folderName)
	fmt.Printf("FileIO:         %v (%.1f%%)\n", timing.FileIO, float64(timing.FileIO)/float64(totalTime)*100)
	fmt.Printf("Preprocessing:  %v (%.1f%%)\n", timing.Preprocessing, float64(timing.Preprocessing)/float64(totalTime)*100)
	fmt.Printf("FeatureDetect:  %v (%.1f%%)\n", timing.FeatureDetect, float64(timing.FeatureDetect)/float64(totalTime)*100)
	fmt.Printf("OpticalFlow:    %v (%.1f%%)\n", timing.OpticalFlow, float64(timing.OpticalFlow)/float64(totalTime)*100)
	fmt.Printf("Visualization:  %v (%.1f%%)\n", timing.Visualization, float64(timing.Visualization)/float64(totalTime)*100)
	fmt.Printf("Total:         %v\n", totalTime)

	return nil
}
