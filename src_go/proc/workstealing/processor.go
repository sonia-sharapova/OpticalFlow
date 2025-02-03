/*
Work Stealing

Includes:
- Initial task distribution
- The work stealing process
- Lock-Free operations

Round Robin for initial task distribution:

	https://stackoverflow.com/questions/7059519/algorithm-for-resource-assignment-in-round-robin-modeled-processor-scheduling
*/
package workstealing

import (
	"fmt"
	"image"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"time"

	"proj3/dicom"
	"proj3/flow"
	"proj3/proc"
	"proj3/viz"
)

type processResult struct {
	frames []image.Image
	timing proc.ProcessTiming
}

type WorkStealingProcessor struct {
	proc.BaseProcessor
	numWorkers int
	deques     []*Deque
	rng        *rand.Rand
}

func NewWorkStealingProcessor(workers int) *WorkStealingProcessor {
	deques := make([]*Deque, workers)
	for i := 0; i < workers; i++ {
		deques[i] = NewDeque()
	}

	return &WorkStealingProcessor{
		BaseProcessor: proc.NewBaseProcessor(),
		numWorkers:    workers,
		deques:        deques,
		rng:           rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (p *WorkStealingProcessor) ProcessDataset(inputDir, outputDir string) error {
	entries, err := filepath.Glob(filepath.Join(inputDir, "*"))
	if err != nil {
		return fmt.Errorf("failed to read directory: %w", err)
	}

	var allTiming proc.ProcessTiming
	start := time.Now()

	// Process each folder
	for _, entry := range entries {
		if info, err := os.Stat(entry); err != nil || !info.IsDir() {
			continue
		}

		// Create output folder
		folderName := filepath.Base(entry)
		outputFolder := filepath.Join(outputDir, folderName)
		if err := os.MkdirAll(outputFolder, 0755); err != nil {
			return fmt.Errorf("failed to create output folder: %w", err)
		}

		// Load files
		t := time.Now()
		files, err := dicom.LoadDICOMFiles(entry)
		if err != nil {
			fmt.Printf("Error loading files from %s: %v\n", entry, err)
			continue
		}
		allTiming.FileIO += time.Since(t)

		if len(files) < 2 {
			continue
		}

		// Create initial tasks
		batchSize := 5 // Process 5 frames per batch
		for i := 0; i < len(files); i += batchSize {
			end := i + batchSize
			if end > len(files) {
				end = len(files)
			}

			task := &Task{
				Files:      files[i:end],
				StartIndex: i,
				OutputPath: outputFolder,
			}

			// Distribute initial tasks round-robin
			workerID := (i / batchSize) % p.numWorkers
			p.deques[workerID].PushBottom(task)
		}

		// Start workers
		var wg sync.WaitGroup
		resultCh := make(chan processResult, p.numWorkers)

		for i := 0; i < p.numWorkers; i++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				frames, timing := p.worker(workerID)
				resultCh <- processResult{frames: frames, timing: timing}
			}(i)
		}

		// Wait for all workers and close result channel
		go func() {
			wg.Wait()
			close(resultCh)
		}()

		// Collect and combine results
		var allFrames []image.Image
		for result := range resultCh {
			allFrames = append(allFrames, result.frames...)
			allTiming.Preprocessing += result.timing.Preprocessing
			allTiming.FeatureDetect += result.timing.FeatureDetect
			allTiming.OpticalFlow += result.timing.OpticalFlow
			allTiming.Visualization += result.timing.Visualization
		}

		// Save final GIF
		if len(allFrames) > 0 {
			outputPath := filepath.Join(outputFolder, folderName+".gif")
			if err := viz.SaveAsGIF(allFrames, outputPath, 500); err != nil {
				fmt.Printf("Error saving GIF for %s: %v\n", folderName, err)
			}
		}
	}

	totalTime := time.Since(start)
	fmt.Printf("\nOverall Timing:\n")
	fmt.Printf("FileIO:         %v (%.1f%%)\n", allTiming.FileIO, float64(allTiming.FileIO)/float64(totalTime)*100)
	fmt.Printf("Preprocessing:  %v (%.1f%%)\n", allTiming.Preprocessing, float64(allTiming.Preprocessing)/float64(totalTime)*100)
	fmt.Printf("FeatureDetect:  %v (%.1f%%)\n", allTiming.FeatureDetect, float64(allTiming.FeatureDetect)/float64(totalTime)*100)
	fmt.Printf("OpticalFlow:    %v (%.1f%%)\n", allTiming.OpticalFlow, float64(allTiming.OpticalFlow)/float64(totalTime)*100)
	fmt.Printf("Visualization:  %v (%.1f%%)\n", allTiming.Visualization, float64(allTiming.Visualization)/float64(totalTime)*100)
	fmt.Printf("Total:         %v\n", totalTime)

	return nil
}

func (p *WorkStealingProcessor) worker(id int) ([]image.Image, proc.ProcessTiming) {
	var timing proc.ProcessTiming
	var frames []image.Image
	myDeque := p.deques[id]

	for {
		// Try to get work from own deque
		task := myDeque.PopBottom()
		if task == nil {
			// Try to steal from other workers
			stolen := false
			attempts := 0
			maxAttempts := p.numWorkers * 2

			for !stolen && attempts < maxAttempts {
				victim := p.rng.Intn(p.numWorkers)
				if victim == id {
					continue
				}

				if task = p.deques[victim].PopTop(); task != nil {
					stolen = true
					break
				}
				attempts++
			}

			if !stolen {
				// No work found after multiple attempts
				return frames, timing
			}
		}

		// Process the task
		taskFrames := p.processTask(task, &timing)
		frames = append(frames, taskFrames...)
	}
}

func (p *WorkStealingProcessor) processTask(task *Task, timing *proc.ProcessTiming) []image.Image {
	var frames []image.Image
	var processedFrames []dicom.FrameData

	// Stage 1: Preprocessing
	t := time.Now()
	for _, file := range task.Files {
		mat, err := dicom.PreprocessDICOM(file, image.Point{X: 256, Y: 256})
		if err != nil {
			continue
		}
		processedFrames = append(processedFrames, dicom.FrameData{Mat: mat})
	}
	timing.Preprocessing += time.Since(t)

	if len(processedFrames) < 2 {
		return frames
	}

	// Stage 2: Feature Detection
	t = time.Now()
	for i := range processedFrames {
		features, err := flow.DetectFeatures(processedFrames[i].Mat, p.GetFeatureParams())
		if err != nil {
			continue
		}
		processedFrames[i].Features = features
	}
	timing.FeatureDetect += time.Since(t)

	// Stage 3: Optical Flow and Visualization
	t = time.Now()
	for i := 1; i < len(processedFrames); i++ {
		prevFrame := processedFrames[i-1]
		currFrame := processedFrames[i]

		nextPoints, status := flow.ComputeOpticalFlow(
			prevFrame.Mat,
			currFrame.Mat,
			prevFrame.Features,
			p.GetFlowParams(),
		)

		var goodPrev, goodNext []image.Point
		for j, tracked := range status {
			if tracked {
				goodPrev = append(goodPrev, prevFrame.Features[j])
				goodNext = append(goodNext, nextPoints[j])
			}
		}

		vizMat := currFrame.Mat.Clone()
		viz.DrawOpticalFlow(&vizMat, goodPrev, goodNext)

		img := viz.MatToImage(vizMat)
		frames = append(frames, img)

		vizMat.Close()
	}
	timing.OpticalFlow += time.Since(t)

	// Cleanup
	for _, frame := range processedFrames {
		frame.Mat.Close()
	}

	return frames
}
