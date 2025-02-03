/*
Pipeline Approach:
Divides the processing into stages that can run concurrently.
	Stage 1: File loading and preprocessing
	Stage 2: Feature detection
	Stage 3: Optical flow computation
	Stage 4: Visualization and GIF creation

Data flows through these stages via channels, allowing multiple frames to be processed simultaneously at different stages.
*/

package proc

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sync"
	"time"

	"proj3/dicom"
	"proj3/flow"
	"proj3/viz"

	"gocv.io/x/gocv"
)

type frameData struct {
	mat        gocv.Mat
	features   []image.Point
	frameIndex int
}

type batchResult struct {
	images []image.Image
	timing ProcessTiming
}

type PipelineProcessor struct {
	BaseProcessor
	numWorkers int
	bufferSize int
	batchSize  int
}

func NewPipelineProcessor(workers, bufferSize int) *PipelineProcessor {
	return &PipelineProcessor{
		BaseProcessor: NewBaseProcessor(),
		numWorkers:    workers,
		bufferSize:    bufferSize,
		batchSize:     5, // Process 5 frames per batch
	}
}

func (p *PipelineProcessor) ProcessDataset(inputDir, outputDir string) error {
	/*
		This processes the entire dataset in parallel using a pipeline of stages:
	*/
	entries, err := filepath.Glob(filepath.Join(inputDir, "*"))
	if err != nil {
		return fmt.Errorf("failed to read directory: %w", err)
	}

	var wg sync.WaitGroup
	for _, entry := range entries {
		if info, err := os.Stat(entry); err != nil || !info.IsDir() {
			continue
		}

		wg.Add(1)
		go func(folder string) {
			defer wg.Done()
			if err := p.processFolder(folder, outputDir); err != nil {
				fmt.Printf("Error processing folder %s: %v\n", filepath.Base(folder), err)
			}
		}(entry)
	}
	wg.Wait()
	return nil
}

func (p *PipelineProcessor) processBatch(files []string, startIdx int, timing *ProcessTiming) ([]image.Image, error) {
	/*
		Here, the function handles a batch of frames through all pipeline stages
	*/
	var frames []frameData
	var result []image.Image

	// Stage 1: Preprocessing all frames in batch
	t := time.Now()
	for i, file := range files {
		mat, err := dicom.PreprocessDICOM(file, image.Point{X: 256, Y: 256})
		if err != nil {
			continue
		}

		frames = append(frames, frameData{
			mat:        mat,
			frameIndex: startIdx + i,
		})
	}
	timing.Preprocessing += time.Since(t)

	if len(frames) < 2 {
		return nil, fmt.Errorf("not enough frames in batch")
	}

	// Stage 2: Feature Detection
	t = time.Now()
	for i := range frames {
		features, err := flow.DetectFeatures(frames[i].mat, p.featureParams)
		if err != nil {
			continue
		}
		frames[i].features = features
	}
	timing.FeatureDetect += time.Since(t)

	// Stage 3: Optical Flow
	t = time.Now()
	for i := 1; i < len(frames); i++ {
		prevFrame := frames[i-1]
		currentFrame := frames[i]

		nextPoints, status := flow.ComputeOpticalFlow(
			prevFrame.mat,
			currentFrame.mat,
			prevFrame.features,
			p.flowParams,
		)

		// Create visualization frame
		var goodPrev, goodNext []image.Point
		for j, tracked := range status {
			if tracked {
				goodPrev = append(goodPrev, prevFrame.features[j])
				goodNext = append(goodNext, nextPoints[j])
			}
		}

		// Create visualization
		vizMat := currentFrame.mat.Clone()
		viz.DrawOpticalFlow(&vizMat, goodPrev, goodNext)

		// Convert to image
		t2 := time.Now()
		img := viz.MatToImage(vizMat)
		timing.Visualization += time.Since(t2)

		result = append(result, img)

		// Cleanup
		vizMat.Close()
	}
	timing.OpticalFlow += time.Since(t)

	// Cleanup
	for _, frame := range frames {
		frame.mat.Close()
	}

	return result, nil
}

func (p *PipelineProcessor) processFolder(inputFolder, outputDir string) error {
	var timing ProcessTiming
	start := time.Now()

	// Create output folder
	folderName := filepath.Base(inputFolder)
	outputFolder := filepath.Join(outputDir, folderName)
	if err := os.MkdirAll(outputFolder, 0755); err != nil {
		return fmt.Errorf("failed to create output folder: %w", err)
	}

	// Load files
	t := time.Now()
	files, err := dicom.LoadDICOMFiles(inputFolder)
	if err != nil {
		return fmt.Errorf("failed to load DICOM files: %w", err)
	}
	timing.FileIO = time.Since(t)

	if len(files) < 2 {
		return fmt.Errorf("not enough frames")
	}

	// Process in batches using worker pool
	var wg sync.WaitGroup
	resultCh := make(chan []image.Image, p.numWorkers)
	batchCh := make(chan []string, (len(files)+p.batchSize-1)/p.batchSize)

	// Create worker pool
	for i := 0; i < p.numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for batch := range batchCh {
				if frames, err := p.processBatch(batch, 0, &timing); err == nil {
					resultCh <- frames
				}
			}
		}()
	}

	// Distribute batches
	for i := 0; i < len(files); i += p.batchSize {
		end := i + p.batchSize
		if end > len(files) {
			end = len(files)
		}
		batchCh <- files[i:end]
	}
	close(batchCh)

	// Collect results
	go func() {
		wg.Wait()
		close(resultCh)
	}()

	// Combine all frames
	var allFrames []image.Image
	for frames := range resultCh {
		allFrames = append(allFrames, frames...)
	}

	// Save final GIF
	if len(allFrames) > 0 {
		outputPath := filepath.Join(outputFolder, folderName+".gif")
		if err := viz.SaveAsGIF(allFrames, outputPath, 500); err != nil {
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
