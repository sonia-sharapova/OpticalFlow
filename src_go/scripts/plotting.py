import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the results
df = pd.read_csv('benchmark_results/results.csv')

# Create a figure with appropriate size
plt.figure(figsize=(12, 6))

# Get sequential baseline for reference line
sequential_time = df[df['Mode'] == 'sequential']['Duration_ms'].iloc[0]
sequential_speedup = df[df['Mode'] == 'sequential']['Speedup'].iloc[0]

# Plot sequential point
plt.scatter(1, sequential_speedup, color='red', marker='*', s=200, label='Sequential', zorder=5)

# Plot for Pipeline mode
pipeline_data = df[df['Mode'] == 'pipeline']
plt.plot(pipeline_data['Workers'], pipeline_data['Speedup'], 
         marker='o', label='Pipeline', linestyle='-')

# Plot for Work-stealing mode
workstealing_data = df[df['Mode'] == 'workstealing']
plt.plot(workstealing_data['Workers'], workstealing_data['Speedup'], 
         marker='s', label='Work-stealing', linestyle='-')

# Add ideal speedup line
max_workers = df['Workers'].max()
plt.plot([1, max_workers], [1, max_workers], 'k--', label='Ideal', alpha=0.5)

# Customize the plot
plt.xlabel('Number of Workers')
plt.ylabel('Speedup')
plt.title('Speedup vs Number of Workers')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

# Ensure sequential point is visible by adjusting axis
plt.xlim(0.5, max_workers + 0.5)
plt.ylim(bottom=0)

# Make the plot tight and save it
plt.tight_layout()
plt.savefig('benchmark_results/speedup.png', bbox_inches='tight', dpi=300)
plt.close()

# Print summary statistics
print("\nSummary Statistics:")
print("\nSequential Mode:")
sequential_data = df[df['Mode'] == 'sequential']
print(sequential_data[['Duration_ms', 'Speedup']].to_string(index=False))

print("\nPipeline Mode:")
print(pipeline_data[['Workers', 'Speedup', 'Duration_ms']].to_string(index=False))

print("\nWork-stealing Mode:")
print(workstealing_data[['Workers', 'Speedup', 'Duration_ms']].to_string(index=False))

# Additional analysis
print("\nMaximum Speedups Achieved:")
for mode in ['pipeline', 'workstealing']:
    mode_data = df[df['Mode'] == mode]
    max_speedup = mode_data['Speedup'].max()
    workers_at_max = mode_data.loc[mode_data['Speedup'].idxmax(), 'Workers']
    print(f"{mode.capitalize()}: {max_speedup:.2f}x at {workers_at_max} workers")