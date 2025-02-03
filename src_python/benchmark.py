import time
import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import optical_flow
import parallel_optical_flow

def run_benchmark(input_dir, folder, num_runs=3):
    """Run benchmark comparing original and parallel implementations."""
    results = {
        'original': [],
        'parallel': []
    }
    
    for run in range(num_runs):
        print(f"\nBenchmark Run {run + 1}/{num_runs}")
        
        # Create temporary output directories for each version
        output_dir_original = f"./benchmark/output/{folder}/original_run_{run}"
        output_dir_parallel = f"./benchmark/output/{folder}/parallel_run_{run}"
        os.makedirs(output_dir_original, exist_ok=True)
        os.makedirs(output_dir_parallel, exist_ok=True)
        
        # Test original version
        print("Testing original implementation...")
        start_time = time.time()
        optical_flow.process_dataset(input_dir, output_dir_original)
        original_time = time.time() - start_time
        results['original'].append(original_time)
        
        # Test parallel version
        print("Testing parallel implementation...")
        start_time = time.time()
        parallel_optical_flow.process_dataset(input_dir, output_dir_parallel)
        parallel_time = time.time() - start_time
        results['parallel'].append(parallel_time)
        
        # Clean up output directories
        shutil.rmtree(output_dir_original)
        shutil.rmtree(output_dir_parallel)
        
        print(f"\nRun {run + 1} Results:")
        print(f"Original Version: {original_time:.2f} seconds")
        print(f"Parallel Version: {parallel_time:.2f} seconds")
        print(f"Speedup: {original_time/parallel_time:.2f}x")

    return results

def plot_results(results, folder):
    """Create visualization of benchmark results."""
    # Calculate statistics
    orig_mean = np.mean(results['original'])
    orig_std = np.std(results['original'])
    par_mean = np.mean(results['parallel'])
    par_std = np.std(results['parallel'])
    speedup = orig_mean / par_mean
    
    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot execution times
    versions = ['Original', 'Parallel']
    times = [orig_mean, par_mean]
    errors = [orig_std, par_std]
    
    ax1.bar(versions, times, yerr=errors, capsize=5)
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Average Execution Time Comparison')
    
    # Add time values on top of bars
    for i, v in enumerate(times):
        ax1.text(i, v + errors[i], f'{v:.2f}s', ha='center', va='bottom')
    
    # Plot individual runs
    runs = range(1, len(results['original']) + 1)
    ax2.plot(runs, results['original'], 'o-', label='Original')
    ax2.plot(runs, results['parallel'], 'o-', label='Parallel')
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('Individual Run Comparison')
    ax2.legend()
    
    # Add overall speedup as text
    fig.suptitle(f'Benchmark Results (Average Speedup: {speedup:.2f}x)', y=1.02)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"./benchmark/results/{folder}", exist_ok=True)
    plt.savefig(f'benchmark/results/{folder}/{timestamp}.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Ensure the input directory exists
    input_dir = sys.argv[1]
    folder_name = input_dir.split("/")[-1]
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Create benchmark output directory
    os.makedirs("./benchmark/output", exist_ok=True)
    os.makedirs("./benchmark/results", exist_ok=True)
    
    print("Starting benchmark...")
    print(f"Input directory: {input_dir}")
    
    # Run benchmarks
    results = run_benchmark(input_dir, folder_name, num_runs=3)
    
    # Calculate and display summary statistics
    orig_mean = np.mean(results['original'])
    par_mean = np.mean(results['parallel'])
    speedup = orig_mean / par_mean
    
    print("\nBenchmark Summary:")
    print(f"Average Original Time: {orig_mean:.2f} seconds")
    print(f"Average Parallel Time: {par_mean:.2f} seconds")
    print(f"Average Speedup: {speedup:.2f}x")
    
    # Create visualization
    plot_results(results, folder_name)
    print("\nBenchmark visualization has been saved.")

if __name__ == "__main__":
    main()