package common

const (
	// MarieClusterLabelKey is the label key for the MarieCluster name
	MarieClusterLabelKey = "marie.ai/cluster"
	// MarieNodeTypeLabelKey is the label key for the node type (server/executor)
	MarieNodeTypeLabelKey = "marie.ai/node-type"
	// MarieNodeGroupLabelKey is the label key for the executor group name
	MarieNodeGroupLabelKey = "marie.ai/group"
	// MarieComponentLabelKey is the label key for the component name
	MarieComponentLabelKey = "marie.ai/component"
	// MarieJobLabelKey is the label key for the MarieJob name
	MarieJobLabelKey = "marie.ai/job"

	// ApplicationName is the application name for Kubernetes labels
	ApplicationName = "marie-ai"
	// ComponentName is the component name for Kubernetes labels
	ComponentName = "marie-operator"

	// KubernetesAppNameLabelKey is the standard Kubernetes app name label
	KubernetesAppNameLabelKey = "app.kubernetes.io/name"
	// KubernetesComponentLabelKey is the standard Kubernetes component label
	KubernetesComponentLabelKey = "app.kubernetes.io/component"
	// KubernetesInstanceLabelKey is the standard Kubernetes instance label
	KubernetesInstanceLabelKey = "app.kubernetes.io/instance"
	// KubernetesManagedByLabelKey is the standard Kubernetes managed-by label
	KubernetesManagedByLabelKey = "app.kubernetes.io/managed-by"

	// NodeTypeServer is the node type for the Marie server
	NodeTypeServer = "server"
	// NodeTypeExecutor is the node type for Marie executors
	NodeTypeExecutor = "executor"
)

// ServerLabels returns the labels for the Marie server pods
func ServerLabels(clusterName string) map[string]string {
	return map[string]string{
		MarieClusterLabelKey:        clusterName,
		MarieNodeTypeLabelKey:       NodeTypeServer,
		MarieComponentLabelKey:      "server",
		KubernetesAppNameLabelKey:   ApplicationName,
		KubernetesInstanceLabelKey:  clusterName,
		KubernetesManagedByLabelKey: ComponentName,
	}
}

// ExecutorLabels returns the labels for executor pods in a specific group
func ExecutorLabels(clusterName, groupName string) map[string]string {
	return map[string]string{
		MarieClusterLabelKey:        clusterName,
		MarieNodeTypeLabelKey:       NodeTypeExecutor,
		MarieNodeGroupLabelKey:      groupName,
		MarieComponentLabelKey:      "executor",
		KubernetesAppNameLabelKey:   ApplicationName,
		KubernetesInstanceLabelKey:  clusterName,
		KubernetesManagedByLabelKey: ComponentName,
	}
}

// JobLabels returns the labels for MarieJob resources
func JobLabels(jobName, clusterName string) map[string]string {
	return map[string]string{
		MarieJobLabelKey:            jobName,
		MarieClusterLabelKey:        clusterName,
		KubernetesAppNameLabelKey:   ApplicationName,
		KubernetesManagedByLabelKey: ComponentName,
	}
}

// ClusterLabels returns the labels for cluster-level resources
func ClusterLabels(clusterName string) map[string]string {
	return map[string]string{
		MarieClusterLabelKey:        clusterName,
		KubernetesAppNameLabelKey:   ApplicationName,
		KubernetesManagedByLabelKey: ComponentName,
	}
}
