package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// MarieJobSpec defines the desired state of MarieJob
type MarieJobSpec struct {
	// ClusterSelector selects an existing MarieCluster to submit the job to
	// +optional
	ClusterSelector map[string]string `json:"clusterSelector,omitempty"`

	// ClusterRef references a MarieCluster by name (alternative to ClusterSelector)
	// +optional
	ClusterRef string `json:"clusterRef,omitempty"`

	// SubmissionMode defines how the job is submitted
	// +kubebuilder:validation:Enum=HTTPSubmission;K8sJobSubmission
	// +kubebuilder:default=HTTPSubmission
	SubmissionMode JobSubmissionMode `json:"submissionMode,omitempty"`

	// JobType is the type of processing job
	// +kubebuilder:validation:Enum=extract;classify;ocr;ner;transform;pipeline
	// +kubebuilder:default=pipeline
	JobType string `json:"jobType,omitempty"`

	// Entrypoint is the processing command or pipeline definition
	// +optional
	Entrypoint string `json:"entrypoint,omitempty"`

	// Input defines the input configuration
	// +optional
	Input *JobInput `json:"input,omitempty"`

	// Output defines the output configuration
	// +optional
	Output *JobOutput `json:"output,omitempty"`

	// Options are job-specific options passed to processors
	// +optional
	Options map[string]string `json:"options,omitempty"`

	// Priority of the job (-100 to 100)
	// +kubebuilder:validation:Minimum=-100
	// +kubebuilder:validation:Maximum=100
	// +kubebuilder:default=0
	Priority int32 `json:"priority,omitempty"`

	// SLA defines service level agreement parameters
	// +optional
	SLA *JobSLA `json:"sla,omitempty"`

	// ActiveDeadlineSeconds is the maximum time the job can run
	// +optional
	ActiveDeadlineSeconds *int64 `json:"activeDeadlineSeconds,omitempty"`

	// BackoffLimit is the number of retries before marking job as failed
	// +kubebuilder:default=3
	BackoffLimit *int32 `json:"backoffLimit,omitempty"`

	// Suspend pauses job execution
	// +optional
	Suspend bool `json:"suspend,omitempty"`

	// ShutdownAfterJobFinishes deletes the cluster after job completion
	// Only applicable when job creates its own cluster
	// +optional
	ShutdownAfterJobFinishes bool `json:"shutdownAfterJobFinishes,omitempty"`

	// TTLSecondsAfterFinished is the time to keep the job after completion
	// +optional
	TTLSecondsAfterFinished *int32 `json:"ttlSecondsAfterFinished,omitempty"`

	// SubmitterPodTemplate is the template for the job submitter pod
	// Used for K8sJobSubmission mode
	// +optional
	SubmitterPodTemplate *corev1.PodTemplateSpec `json:"submitterPodTemplate,omitempty"`
}

// JobSubmissionMode defines how jobs are submitted
type JobSubmissionMode string

const (
	// HTTPSubmission submits jobs via HTTP API to the Marie server
	HTTPSubmission JobSubmissionMode = "HTTPSubmission"
	// K8sJobSubmission creates a Kubernetes Job to submit and monitor the job
	K8sJobSubmission JobSubmissionMode = "K8sJobSubmission"
)

// JobInput defines input configuration for a job
type JobInput struct {
	// URI is the input location (s3://, file://, etc.)
	// +optional
	URI string `json:"uri,omitempty"`

	// Data is inline input data (base64 encoded for binary)
	// +optional
	Data string `json:"data,omitempty"`

	// ConfigMapRef references a ConfigMap containing input data
	// +optional
	ConfigMapRef *corev1.LocalObjectReference `json:"configMapRef,omitempty"`

	// SecretRef references a Secret containing input data
	// +optional
	SecretRef *corev1.LocalObjectReference `json:"secretRef,omitempty"`

	// ContentType is the MIME type of the input
	// +kubebuilder:default="application/pdf"
	ContentType string `json:"contentType,omitempty"`
}

// JobOutput defines output configuration for a job
type JobOutput struct {
	// URI is the output location (s3://, file://, etc.)
	// +optional
	URI string `json:"uri,omitempty"`

	// Format is the output format
	// +kubebuilder:validation:Enum=json;xml;csv;yaml
	// +kubebuilder:default=json
	Format string `json:"format,omitempty"`

	// ConfigMapRef to store output as ConfigMap
	// +optional
	ConfigMapRef *corev1.LocalObjectReference `json:"configMapRef,omitempty"`
}

// JobSLA defines service level agreement parameters
type JobSLA struct {
	// SoftDeadline is the preferred completion time (e.g., "5m", "1h")
	// +optional
	SoftDeadline string `json:"softDeadline,omitempty"`

	// HardDeadline is the maximum completion time before job is cancelled
	// +optional
	HardDeadline string `json:"hardDeadline,omitempty"`
}

// MarieJobStatus defines the observed state of MarieJob
type MarieJobStatus struct {
	// Phase is the current phase of the job
	// +optional
	Phase JobPhase `json:"phase,omitempty"`

	// Conditions represent the latest available observations
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// DagID is the internal DAG ID from Marie scheduler
	// +optional
	DagID string `json:"dagId,omitempty"`

	// JobID is the internal job ID
	// +optional
	JobID string `json:"jobId,omitempty"`

	// Message provides additional status information
	// +optional
	Message string `json:"message,omitempty"`

	// StartTime is when the job started
	// +optional
	StartTime *metav1.Time `json:"startTime,omitempty"`

	// CompletionTime is when the job completed
	// +optional
	CompletionTime *metav1.Time `json:"completionTime,omitempty"`

	// Duration is the job duration
	// +optional
	Duration string `json:"duration,omitempty"`

	// Progress is the job progress percentage (0-100)
	// +optional
	Progress int32 `json:"progress,omitempty"`

	// RetryCount is the number of retries attempted
	// +optional
	RetryCount int32 `json:"retryCount,omitempty"`

	// Result contains the job result
	// +optional
	Result *JobResult `json:"result,omitempty"`

	// Error contains error information if job failed
	// +optional
	Error *JobError `json:"error,omitempty"`

	// ClusterName is the name of the cluster running this job
	// +optional
	ClusterName string `json:"clusterName,omitempty"`

	// SubmitterJobName is the name of the K8s Job for submission
	// +optional
	SubmitterJobName string `json:"submitterJobName,omitempty"`

	// ObservedGeneration is the most recent generation observed
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
}

// JobPhase represents the phase of a MarieJob
// +kubebuilder:validation:Enum=Pending;Initializing;Running;Succeeded;Failed;Cancelled
type JobPhase string

const (
	JobPhasePending      JobPhase = "Pending"
	JobPhaseInitializing JobPhase = "Initializing"
	JobPhaseRunning      JobPhase = "Running"
	JobPhaseSucceeded    JobPhase = "Succeeded"
	JobPhaseFailed       JobPhase = "Failed"
	JobPhaseCancelled    JobPhase = "Cancelled"
)

// JobResult contains the result of a completed job
type JobResult struct {
	// OutputURI is the location of the output
	// +optional
	OutputURI string `json:"outputUri,omitempty"`

	// Summary contains a summary of the processing results
	// +optional
	Summary map[string]string `json:"summary,omitempty"`

	// DocumentCount is the number of documents processed
	// +optional
	DocumentCount int32 `json:"documentCount,omitempty"`

	// PageCount is the total number of pages processed
	// +optional
	PageCount int32 `json:"pageCount,omitempty"`
}

// JobError contains error information
type JobError struct {
	// Code is the error code
	Code string `json:"code,omitempty"`
	// Message is the error message
	Message string `json:"message,omitempty"`
	// Details contains additional error details
	// +optional
	Details string `json:"details,omitempty"`
}

// Condition types for MarieJob
const (
	// JobSubmitted indicates the job was submitted
	JobSubmitted = "JobSubmitted"
	// JobRunning indicates the job is running
	JobRunning = "JobRunning"
	// JobComplete indicates the job completed
	JobComplete = "JobComplete"
)

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=mj
// +kubebuilder:printcolumn:name="Cluster",type="string",JSONPath=".status.clusterName"
// +kubebuilder:printcolumn:name="Type",type="string",JSONPath=".spec.jobType"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Progress",type="integer",JSONPath=".status.progress"
// +kubebuilder:printcolumn:name="Duration",type="string",JSONPath=".status.duration"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// MarieJob is the Schema for the mariejobs API
type MarieJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   MarieJobSpec   `json:"spec,omitempty"`
	Status MarieJobStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// MarieJobList contains a list of MarieJob
type MarieJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []MarieJob `json:"items"`
}

func init() {
	SchemeBuilder.Register(&MarieJob{}, &MarieJobList{})
}
