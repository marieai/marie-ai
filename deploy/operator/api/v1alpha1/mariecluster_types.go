package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// MarieClusterSpec defines the desired state of MarieCluster
type MarieClusterSpec struct {
	// ServerSpec defines the Marie server (gateway + scheduler) configuration
	// +kubebuilder:validation:Required
	ServerSpec ServerGroupSpec `json:"serverSpec"`

	// ExecutorGroupSpecs defines the executor worker pools
	// +optional
	ExecutorGroupSpecs []ExecutorGroupSpec `json:"executorGroupSpecs,omitempty"`

	// Suspend suspends all pods without deleting them
	// +optional
	Suspend *bool `json:"suspend,omitempty"`

	// EnableIngress enables ingress for the server
	// +optional
	EnableIngress *bool `json:"enableIngress,omitempty"`

	// ServiceType for the server service
	// +kubebuilder:validation:Enum=ClusterIP;NodePort;LoadBalancer
	// +kubebuilder:default=ClusterIP
	ServiceType corev1.ServiceType `json:"serviceType,omitempty"`

	// MarieVersion specifies the Marie-AI version
	// +optional
	MarieVersion string `json:"marieVersion,omitempty"`
}

// ServerGroupSpec defines the Marie server configuration
type ServerGroupSpec struct {
	// Replicas is the number of server replicas
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	Replicas *int32 `json:"replicas,omitempty"`

	// Template is the pod template for the server
	// +kubebuilder:validation:Required
	Template corev1.PodTemplateSpec `json:"template"`

	// ServiceType overrides the cluster-level service type
	// +optional
	ServiceType corev1.ServiceType `json:"serviceType,omitempty"`

	// ServerStartParams are additional CLI arguments for marie server
	// +optional
	ServerStartParams map[string]string `json:"serverStartParams,omitempty"`

	// ConfigFile path to marie.yml configuration
	// +kubebuilder:default="/config/marie.yml"
	ConfigFile string `json:"configFile,omitempty"`
}

// ExecutorGroupSpec defines an executor worker pool
type ExecutorGroupSpec struct {
	// GroupName is the unique name of this executor group
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Pattern=`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`
	GroupName string `json:"groupName"`

	// Replicas is the number of executor replicas in this group
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=0
	Replicas *int32 `json:"replicas,omitempty"`

	// MinReplicas for autoscaling (0 enables scale-to-zero)
	// +optional
	MinReplicas *int32 `json:"minReplicas,omitempty"`

	// MaxReplicas for autoscaling
	// +optional
	MaxReplicas *int32 `json:"maxReplicas,omitempty"`

	// Template is the pod template for executors in this group
	// +kubebuilder:validation:Required
	Template corev1.PodTemplateSpec `json:"template"`

	// ExecutorType identifies the type of executor (e.g., "ocr", "classifier", "ner")
	// +optional
	ExecutorType string `json:"executorType,omitempty"`

	// GPU indicates this group requires GPU resources
	// +optional
	GPU bool `json:"gpu,omitempty"`

	// ScaleStrategy defines how to handle scale-down
	// +optional
	ScaleStrategy ScaleStrategy `json:"scaleStrategy,omitempty"`

	// Suspend suspends this specific group
	// +optional
	Suspend *bool `json:"suspend,omitempty"`
}

// ScaleStrategy defines the scale-down strategy for executor groups
type ScaleStrategy struct {
	// WorkersToDelete lists specific pod names to delete during scale-down
	// +optional
	WorkersToDelete []string `json:"workersToDelete,omitempty"`
}

// MarieClusterStatus defines the observed state of MarieCluster
type MarieClusterStatus struct {
	// Phase represents the current phase of the cluster
	// +optional
	Phase ClusterPhase `json:"phase,omitempty"`

	// Conditions represent the latest available observations
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// ServerReady indicates if the server is ready
	// +optional
	ServerReady bool `json:"serverReady,omitempty"`

	// ReadyExecutorReplicas is the total number of ready executor replicas
	// +optional
	ReadyExecutorReplicas int32 `json:"readyExecutorReplicas,omitempty"`

	// AvailableExecutorReplicas is the total number of available executor replicas
	// +optional
	AvailableExecutorReplicas int32 `json:"availableExecutorReplicas,omitempty"`

	// DesiredExecutorReplicas is the total desired executor replicas
	// +optional
	DesiredExecutorReplicas int32 `json:"desiredExecutorReplicas,omitempty"`

	// ExecutorGroupStatuses contains status for each executor group
	// +optional
	ExecutorGroupStatuses map[string]ExecutorGroupStatus `json:"executorGroupStatuses,omitempty"`

	// DesiredCPU is the total desired CPU across all pods
	// +optional
	DesiredCPU resource.Quantity `json:"desiredCPU,omitempty"`

	// DesiredMemory is the total desired memory across all pods
	// +optional
	DesiredMemory resource.Quantity `json:"desiredMemory,omitempty"`

	// DesiredGPU is the total desired GPU across all pods
	// +optional
	DesiredGPU resource.Quantity `json:"desiredGPU,omitempty"`

	// Endpoints contains the cluster endpoints
	// +optional
	Endpoints map[string]string `json:"endpoints,omitempty"`

	// ServerInfo contains information about the server pod
	// +optional
	ServerInfo ServerInfo `json:"serverInfo,omitempty"`

	// ObservedGeneration is the most recent generation observed
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// LastUpdateTime is the last time the status was updated
	// +optional
	LastUpdateTime *metav1.Time `json:"lastUpdateTime,omitempty"`
}

// ClusterPhase represents the phase of a MarieCluster
// +kubebuilder:validation:Enum=Pending;Creating;Running;Suspended;Failed;Terminating
type ClusterPhase string

const (
	ClusterPhasePending     ClusterPhase = "Pending"
	ClusterPhaseCreating    ClusterPhase = "Creating"
	ClusterPhaseRunning     ClusterPhase = "Running"
	ClusterPhaseSuspended   ClusterPhase = "Suspended"
	ClusterPhaseFailed      ClusterPhase = "Failed"
	ClusterPhaseTerminating ClusterPhase = "Terminating"
)

// ExecutorGroupStatus contains status for an executor group
type ExecutorGroupStatus struct {
	// Replicas is the number of replicas
	Replicas int32 `json:"replicas,omitempty"`
	// ReadyReplicas is the number of ready replicas
	ReadyReplicas int32 `json:"readyReplicas,omitempty"`
	// AvailableReplicas is the number of available replicas
	AvailableReplicas int32 `json:"availableReplicas,omitempty"`
}

// ServerInfo contains information about the server
type ServerInfo struct {
	// PodName is the name of the server pod
	PodName string `json:"podName,omitempty"`
	// PodIP is the IP address of the server pod
	PodIP string `json:"podIP,omitempty"`
	// ServiceName is the name of the server service
	ServiceName string `json:"serviceName,omitempty"`
}

// Condition types for MarieCluster
const (
	// MarieClusterProvisioned indicates the cluster resources are created
	MarieClusterProvisioned = "MarieClusterProvisioned"
	// ServerPodReady indicates the server pod is ready
	ServerPodReady = "ServerPodReady"
	// AllExecutorsReady indicates all executor pods are ready
	AllExecutorsReady = "AllExecutorsReady"
)

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=mc;marie
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Server",type="boolean",JSONPath=".status.serverReady"
// +kubebuilder:printcolumn:name="Executors",type="integer",JSONPath=".status.readyExecutorReplicas"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// MarieCluster is the Schema for the marieclusters API
type MarieCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   MarieClusterSpec   `json:"spec,omitempty"`
	Status MarieClusterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// MarieClusterList contains a list of MarieCluster
type MarieClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []MarieCluster `json:"items"`
}

func init() {
	SchemeBuilder.Register(&MarieCluster{}, &MarieClusterList{})
}
