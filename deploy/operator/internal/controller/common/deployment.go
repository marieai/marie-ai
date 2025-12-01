package common

import (
	"context"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	ctrl "sigs.k8s.io/controller-runtime"

	mariav1alpha1 "github.com/marieai/marie-ai/deploy/operator/api/v1alpha1"
)

// BuildServerDeployment builds the Deployment for the Marie server
func BuildServerDeployment(ctx context.Context, instance *mariav1alpha1.MarieCluster) *appsv1.Deployment {
	log := ctrl.LoggerFrom(ctx)
	log.Info("Building server deployment", "cluster", instance.Name)

	labels := ServerLabels(instance.Name)
	podTemplate := BuildServerPodTemplateSpec(ctx, instance)

	replicas := int32(1)
	if instance.Spec.ServerSpec.Replicas != nil {
		replicas = *instance.Spec.ServerSpec.Replicas
	}

	// Check if suspended
	if instance.Spec.Suspend != nil && *instance.Spec.Suspend {
		replicas = 0
	}

	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      GenerateServerDeploymentName(instance.Name),
			Namespace: instance.Namespace,
			Labels:    labels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					MarieClusterLabelKey:  instance.Name,
					MarieNodeTypeLabelKey: NodeTypeServer,
				},
			},
			Template: podTemplate,
			Strategy: appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &appsv1.RollingUpdateDeployment{
					MaxUnavailable: intOrStrPtr(0),
					MaxSurge:       intOrStrPtr(1),
				},
			},
		},
	}

	return deployment
}

// BuildExecutorDeployment builds the Deployment for an executor group
func BuildExecutorDeployment(ctx context.Context, instance *mariav1alpha1.MarieCluster, groupSpec *mariav1alpha1.ExecutorGroupSpec) *appsv1.Deployment {
	log := ctrl.LoggerFrom(ctx)
	log.Info("Building executor deployment", "cluster", instance.Name, "group", groupSpec.GroupName)

	labels := ExecutorLabels(instance.Name, groupSpec.GroupName)
	podTemplate := BuildExecutorPodTemplateSpec(ctx, instance, groupSpec)

	replicas := int32(1)
	if groupSpec.Replicas != nil {
		replicas = *groupSpec.Replicas
	}

	// Check if cluster is suspended
	if instance.Spec.Suspend != nil && *instance.Spec.Suspend {
		replicas = 0
	}
	// Check if executor group is suspended
	if groupSpec.Suspend != nil && *groupSpec.Suspend {
		replicas = 0
	}

	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      GenerateExecutorDeploymentName(instance.Name, groupSpec.GroupName),
			Namespace: instance.Namespace,
			Labels:    labels,
			Annotations: map[string]string{
				"marie.ai/executor-group": groupSpec.GroupName,
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					MarieClusterLabelKey:   instance.Name,
					MarieNodeTypeLabelKey:  NodeTypeExecutor,
					MarieNodeGroupLabelKey: groupSpec.GroupName,
				},
			},
			Template: podTemplate,
			Strategy: appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &appsv1.RollingUpdateDeployment{
					MaxUnavailable: intOrStrPtr(0),
					MaxSurge:       intOrStrPtr(1),
				},
			},
		},
	}

	// Add GPU annotations for scale-to-zero if this is a GPU group
	if groupSpec.GPU {
		if deployment.Annotations == nil {
			deployment.Annotations = make(map[string]string)
		}
		// Annotation for cluster autoscaler to understand GPU requirements
		deployment.Annotations["cluster-autoscaler.kubernetes.io/safe-to-evict"] = "true"
	}

	// Add annotations for scale-to-zero support
	if groupSpec.MinReplicas != nil && *groupSpec.MinReplicas == 0 {
		if deployment.Annotations == nil {
			deployment.Annotations = make(map[string]string)
		}
		deployment.Annotations["marie.ai/scale-to-zero"] = "enabled"
	}

	return deployment
}

// intOrStrPtr returns a pointer to an IntOrString value
func intOrStrPtr(val int) *intstr.IntOrString {
	v := intstr.FromInt(val)
	return &v
}
