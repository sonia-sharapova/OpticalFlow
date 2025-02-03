package proc

import (
	"proj3/flow"
)

// ExecutionMode defines the type of processing to use
type ExecutionMode string

const (
	Sequential   ExecutionMode = "sequential"
	Pipeline     ExecutionMode = "pipeline"
	WorkStealing ExecutionMode = "workstealing"
	Hybrid       ExecutionMode = "hybrid"
)

// ProcessingOptions contains all configuration options for processing
type ProcessingOptions struct {
	InputDir   string
	OutputDir  string
	Workers    int
	Mode       ExecutionMode
	BufferSize int
}

// FolderData represents a directory of DICOM files to be processed
type FolderData struct {
	FolderPath string
	OutputPath string
	Files      []string
}

// Processor interface defines the common interface for all processing implementations
type Processor interface {
	ProcessDataset(inputDir, outputDir string) error
}

// BaseProcessor contains common fields for all processors
type BaseProcessor struct {
	flowParams    flow.OpticalFlowParams
	featureParams flow.ShiTomasiParams
}

// NewBaseProcessor creates a new base processor with default parameters
func NewBaseProcessor() BaseProcessor {
	return BaseProcessor{
		flowParams:    flow.DefaultOpticalFlowParams(),
		featureParams: flow.DefaultShiTomasiParams(),
	}
}

// GetFlowParams returns the optical flow parameters
func (b BaseProcessor) GetFlowParams() flow.OpticalFlowParams {
	return b.flowParams
}

// GetFeatureParams returns the feature detection parameters
func (b BaseProcessor) GetFeatureParams() flow.ShiTomasiParams {
	return b.featureParams
}
