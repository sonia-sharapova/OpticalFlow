package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"

	"proj3/benchmark"
)

func main() {
	// Define command-line flags
	inputDir := flag.String("input", "", "Input directory containing DICOM files")
	maxWorkers := flag.Int("maxworkers", runtime.NumCPU(), "Maximum number of workers to test")
	outputDir := flag.String("output", "benchmark_results", "Output directory for results")
	help := flag.Bool("help", false, "Show usage information")

	flag.Parse()

	if *help || *inputDir == "" {
		printUsage()
		os.Exit(0)
	}

	// Generate worker counts from 2 to maxWorkers
	workerCounts := make([]int, 0)
	for w := 2; w <= *maxWorkers; w += 2 {
		if w > 12 { // Cap at 12 workers
			break
		}
		workerCounts = append(workerCounts, w)
	}

	fmt.Printf("Starting benchmarks with worker counts: %v\n", workerCounts)
	fmt.Printf("Input directory: %s\n", *inputDir)

	// Create output directory if it doesn't exist
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatal("Failed to create output directory:", err)
	}

	// Run benchmarks
	if err := benchmark.RunExperiments(*inputDir, workerCounts); err != nil {
		log.Fatal("Benchmark failed:", err)
	}

	fmt.Println("\nBenchmarks complete. Results written to benchmark_results/results.csv")
	fmt.Println("Run 'python scripts/plot_results.py' to generate speedup graphs")
}

func printUsage() {
	fmt.Println("DICOM Optical Flow Processor Benchmark")
	fmt.Println("\nUsage:")
	fmt.Printf("  %s [options]\n", os.Args[0])
	fmt.Println("\nOptions:")
	flag.PrintDefaults()
	fmt.Println("\nExample:")
	fmt.Printf("  %s -input ./data -maxworkers 8\n", os.Args[0])
	fmt.Println("\nThe program will:")
	fmt.Println("1. Run sequential processing as baseline")
	fmt.Println("2. Test pipeline implementation with 2,4,6,... workers up to maxworkers")
	fmt.Println("3. Test work-stealing implementation with same worker counts")
	fmt.Println("4. Save results to benchmark_results/results.csv")
}
