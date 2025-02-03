# OpticalFlow
Optical Flow to Capture Cardiac Motion (with parallelizations)

### Code Overview
The core functionality includes:
1. Loading and preprocessing DICOM files.
2. Feature detection using Shi-Tomasi corner detection
3. Optical flow computation using the Lucas-Kanade method
4. Parallelization to speed up execution
5. Visualization of the motion analysis as animated GIFs


## Python Implementation:

**Sample data can be found in /src_python/data/smallerData**

#### Install libraries:
    $ pip install pandas matplotlib seaborn
    $ pip install pydicom
    $ pip install pydicom matplotlib
    $ pip install opencv-python-headless numpy

### Files:
- notebook/final.py: contains the code in notebook format (not parallelized)
- src_python/optical_flow.py: sequential optical flow program
- src_python/parallel_optical_flow.py: parallel optical flow program
- src_python/benchmark.py: benchmarking for all performances
- data/smallerData.zip: uzip this to run

### Usage:

#### Independent Runs:
**Sequential:**

\$ python src_python/optical_flow.py data/smallerData

**Parallel:**

\$ python src_python/parallel_optical_flow.py data/smallerData

**Run All (Benchmarking)**

\$ python src_python/benchmark.py data/smallerData



## Go Implementation:

**Sample data can be found in /src_go/data/smallerData**

### Usage:
#### To Run:
Initialize go module:
    go mod init proj3
Install required dependencies:
	go get -u gocv.io/x/gocv 
	go get -u github.com/suyashkumar/dicom 
	go mod tidy

For cluster:
    sbatch ./benchmark/run_benchmark.sh

#### Independent Runs:
**Sequential:**
\$ go run ./cmdprocess-dicom/main.go -input ./smallerData -output ./output/sequential/ -mode sequential 

**Pipeline:**
\$ go run ./cmd/process-dicom/main.go -input ./smallerData -output ./output/pipeline/ -mode pipeline -workers 8 -buffer 10 

**Work Stealing:**
\$ go run ./cmd/process-dicom/main.go -input ./smallerData -output ./output/workstealing/ -mode workstealing -workers 8

**Run All (Benchmarking)**
\$ go run ./cmd/benchmark/main.go -input ./smallerData -maxworkers 8


### Resources:

Motion Estimation with Optical Flow:
	- https://nanonets.com/blog/optical-flow/

Optical Flow in OpenCV:
    - https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

Feature Detection Using Shi-Tomasi Corner Detection:
    - https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html

Feature Detection Implementation:
    - https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
