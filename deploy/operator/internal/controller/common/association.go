package common

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	mariav1alpha1 "github.com/marieai/marie-ai/deploy/operator/api/v1alpha1"
)

// AssociationOption is an interface that can be used for both List and DeleteAllOf operations
type AssociationOption interface {
	client.ListOption
	client.DeleteAllOfOption
}

// AssociationOptions is a slice of AssociationOption
type AssociationOptions []AssociationOption

// ToListOptions converts AssociationOptions to a slice of client.ListOption
func (list AssociationOptions) ToListOptions() []client.ListOption {
	options := make([]client.ListOption, len(list))
	for i, option := range list {
		options[i] = option.(client.ListOption)
	}
	return options
}

// ToDeleteOptions converts AssociationOptions to a slice of client.DeleteAllOfOption
func (list AssociationOptions) ToDeleteOptions() []client.DeleteAllOfOption {
	options := make([]client.DeleteAllOfOption, len(list))
	for i, option := range list {
		options[i] = option.(client.DeleteAllOfOption)
	}
	return options
}

// ToMetaV1ListOptions converts AssociationOptions to metav1.ListOptions
func (list AssociationOptions) ToMetaV1ListOptions() metav1.ListOptions {
	listOptions := client.ListOptions{}
	for _, option := range list {
		option.(client.ListOption).ApplyToList(&listOptions)
	}
	return *listOptions.AsListOptions()
}

// ClusterServerPodsAssociationOptions returns options to filter server pods for a cluster
func ClusterServerPodsAssociationOptions(instance *mariav1alpha1.MarieCluster) AssociationOptions {
	return AssociationOptions{
		client.InNamespace(instance.Namespace),
		client.MatchingLabels{
			MarieClusterLabelKey:  instance.Name,
			MarieNodeTypeLabelKey: NodeTypeServer,
		},
	}
}

// ClusterExecutorPodsAssociationOptions returns options to filter all executor pods for a cluster
func ClusterExecutorPodsAssociationOptions(instance *mariav1alpha1.MarieCluster) AssociationOptions {
	return AssociationOptions{
		client.InNamespace(instance.Namespace),
		client.MatchingLabels{
			MarieClusterLabelKey:  instance.Name,
			MarieNodeTypeLabelKey: NodeTypeExecutor,
		},
	}
}

// ClusterExecutorGroupPodsAssociationOptions returns options to filter executor pods in a specific group
func ClusterExecutorGroupPodsAssociationOptions(instance *mariav1alpha1.MarieCluster, groupName string) AssociationOptions {
	return AssociationOptions{
		client.InNamespace(instance.Namespace),
		client.MatchingLabels{
			MarieClusterLabelKey:   instance.Name,
			MarieNodeTypeLabelKey:  NodeTypeExecutor,
			MarieNodeGroupLabelKey: groupName,
		},
	}
}

// ClusterAllPodsAssociationOptions returns options to filter all pods for a cluster
func ClusterAllPodsAssociationOptions(instance *mariav1alpha1.MarieCluster) AssociationOptions {
	return AssociationOptions{
		client.InNamespace(instance.Namespace),
		client.MatchingLabels{
			MarieClusterLabelKey: instance.Name,
		},
	}
}

// ClusterServicesAssociationOptions returns options to filter all services for a cluster
func ClusterServicesAssociationOptions(instance *mariav1alpha1.MarieCluster) AssociationOptions {
	return AssociationOptions{
		client.InNamespace(instance.Namespace),
		client.MatchingLabels{
			MarieClusterLabelKey: instance.Name,
		},
	}
}

// ServerServiceNamespacedName returns the namespaced name for the server service
func ServerServiceNamespacedName(instance *mariav1alpha1.MarieCluster) types.NamespacedName {
	return types.NamespacedName{
		Namespace: instance.Namespace,
		Name:      GenerateServerServiceName(instance.Name),
	}
}

// ServerHeadlessServiceNamespacedName returns the namespaced name for the headless server service
func ServerHeadlessServiceNamespacedName(instance *mariav1alpha1.MarieCluster) types.NamespacedName {
	return types.NamespacedName{
		Namespace: instance.Namespace,
		Name:      GenerateServerHeadlessServiceName(instance.Name),
	}
}

// ExecutorServiceNamespacedName returns the namespaced name for an executor group service
func ExecutorServiceNamespacedName(instance *mariav1alpha1.MarieCluster, groupName string) types.NamespacedName {
	return types.NamespacedName{
		Namespace: instance.Namespace,
		Name:      GenerateExecutorServiceName(instance.Name, groupName),
	}
}

// GenerateServerServiceName generates the server service name
func GenerateServerServiceName(clusterName string) string {
	return clusterName + "-server-svc"
}

// GenerateServerHeadlessServiceName generates the headless server service name
func GenerateServerHeadlessServiceName(clusterName string) string {
	return clusterName + "-server-headless"
}

// GenerateExecutorServiceName generates the executor group service name
func GenerateExecutorServiceName(clusterName, groupName string) string {
	return clusterName + "-executor-" + groupName + "-svc"
}

// GenerateServerDeploymentName generates the server deployment name
func GenerateServerDeploymentName(clusterName string) string {
	return clusterName + "-server"
}

// GenerateExecutorDeploymentName generates the executor deployment name
func GenerateExecutorDeploymentName(clusterName, groupName string) string {
	return clusterName + "-executor-" + groupName
}
