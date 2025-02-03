package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"proj3/proc"
	"proj3/proc/workstealing"
)

func main() {
	// Define command line flags
	inputDir := flag.String("input", "", "Input directory containing DICOM files")
	outputDir := flag.String("output", "", "Output directory for GIF results")
	workers := flag.Int("workers", 4, "Number of worker goroutines")
	mode := flag.String("mode", "sequential", "Execution mode: sequential, pipeline, workstealing, hybrid")
	bufferSize := flag.Int("buffer", 10, "Size of pipeline buffers")
	help := flag.Bool("help", false, "Show usage information")

	flag.Parse()

	if *help || *inputDir == "" || *outputDir == "" {
		printUsage()
		os.Exit(0)
	}

	execMode := proc.ExecutionMode(*mode)
	if !isValidMode(execMode) {
		log.Fatalf("Invalid execution mode: %s", *mode)
	}

	opts := proc.ProcessingOptions{
		InputDir:   *inputDir,
		OutputDir:  *outputDir,
		Workers:    *workers,
		Mode:       execMode,
		BufferSize: *bufferSize,
	}

	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatal("Failed to create output directory:", err)
	}

	if err := processDataset(opts); err != nil {
		log.Fatal("Failed to process dataset:", err)
	}
}

func processDataset(opts proc.ProcessingOptions) error {
	switch opts.Mode {
	case proc.Sequential:
		processor := proc.NewSequentialProcessor()
		return processor.ProcessDataset(opts.InputDir, opts.OutputDir)
	case proc.Pipeline:
		processor := proc.NewPipelineProcessor(opts.Workers, opts.BufferSize)
		return processor.ProcessDataset(opts.InputDir, opts.OutputDir)
	case proc.WorkStealing:
		processor := workstealing.NewWorkStealingProcessor(opts.Workers)
		return processor.ProcessDataset(opts.InputDir, opts.OutputDir)
	case proc.Hybrid:
		return fmt.Errorf("hybrid mode not yet implemented")
	default:
		return fmt.Errorf("unknown execution mode: %s", opts.Mode)
	}
}

func isValidMode(mode proc.ExecutionMode) bool {
	validModes := map[proc.ExecutionMode]bool{
		proc.Sequential:   true,
		proc.Pipeline:     true,
		proc.WorkStealing: true,
		proc.Hybrid:       true,
	}
	return validModes[mode]
}

func printUsage() {
	fmt.Println("DICOM Optical Flow Processor")
	fmt.Println("\nUsage:")
	fmt.Println("  program [options]")
	fmt.Println("\nOptions:")
	flag.PrintDefaults()
	fmt.Println("\nExecution Modes:")
	fmt.Println("  sequential   - Process files sequentially (default)")
	fmt.Println("  pipeline     - Use pipeline parallelism")
	fmt.Println("  workstealing - Use work-stealing parallelism")
	fmt.Println("  hybrid       - Combine pipeline and work-stealing")
}
