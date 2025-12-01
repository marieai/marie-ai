package common

import (
	"context"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"

	mariav1alpha1 "github.com/marieai/marie-ai/deploy/operator/api/v1alpha1"
)

const (
	// MarieContainerName is the name of the main Marie container
	MarieContainerName = "marie"
	// DefaultMarieImage is the default Marie-AI image
	DefaultMarieImage = "marieai/marie:latest"
	// DefaultConfigPath is the default path to the Marie configuration file
	DefaultConfigPath = "/config/marie.yml"
)

// BuildServerPodTemplateSpec builds the PodTemplateSpec for the Marie server
func BuildServerPodTemplateSpec(ctx context.Context, instance *mariav1alpha1.MarieCluster) corev1.PodTemplateSpec {
	log := ctrl.LoggerFrom(ctx)
	log.Info("Building server pod template", "cluster", instance.Name)

	// Start with the user-provided template
	podTemplate := instance.Spec.ServerSpec.Template.DeepCopy()

	// Ensure labels are set
	if podTemplate.Labels == nil {
		podTemplate.Labels = make(map[string]string)
	}
	for k, v := range ServerLabels(instance.Name) {
		podTemplate.Labels[k] = v
	}

	// Add Marie-specific environment variables to the first container
	if len(podTemplate.Spec.Containers) > 0 {
		container := &podTemplate.Spec.Containers[0]
		container.Env = appendEnvVarsIfNotExist(container.Env, getServerEnvVars(instance))
	}

	return *podTemplate
}

// BuildExecutorPodTemplateSpec builds the PodTemplateSpec for an executor group
func BuildExecutorPodTemplateSpec(ctx context.Context, instance *mariav1alpha1.MarieCluster, groupSpec *mariav1alpha1.ExecutorGroupSpec) corev1.PodTemplateSpec {
	log := ctrl.LoggerFrom(ctx)
	log.Info("Building executor pod template", "cluster", instance.Name, "group", groupSpec.GroupName)

	// Start with the user-provided template
	podTemplate := groupSpec.Template.DeepCopy()

	// Ensure labels are set
	if podTemplate.Labels == nil {
		podTemplate.Labels = make(map[string]string)
	}
	for k, v := range ExecutorLabels(instance.Name, groupSpec.GroupName) {
		podTemplate.Labels[k] = v
	}

	// Add Marie-specific environment variables to the first container
	if len(podTemplate.Spec.Containers) > 0 {
		container := &podTemplate.Spec.Containers[0]
		container.Env = appendEnvVarsIfNotExist(container.Env, getExecutorEnvVars(instance, groupSpec))
	}

	return *podTemplate
}

// getServerEnvVars returns the environment variables for the server container
func getServerEnvVars(instance *mariav1alpha1.MarieCluster) []corev1.EnvVar {
	envVars := []corev1.EnvVar{
		{
			Name:  "MARIE_CLUSTER_NAME",
			Value: instance.Name,
		},
		{
			Name: "MARIE_POD_NAME",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "metadata.name",
				},
			},
		},
		{
			Name: "MARIE_POD_NAMESPACE",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "metadata.namespace",
				},
			},
		},
		{
			Name: "MARIE_POD_IP",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "status.podIP",
				},
			},
		},
		{
			Name:  "MARIE_NODE_TYPE",
			Value: NodeTypeServer,
		},
	}

	// Add config file path if specified
	configFile := instance.Spec.ServerSpec.ConfigFile
	if configFile == "" {
		configFile = DefaultConfigPath
	}
	envVars = append(envVars, corev1.EnvVar{
		Name:  "MARIE_CONFIG_FILE",
		Value: configFile,
	})

	return envVars
}

// getExecutorEnvVars returns the environment variables for an executor container
func getExecutorEnvVars(instance *mariav1alpha1.MarieCluster, groupSpec *mariav1alpha1.ExecutorGroupSpec) []corev1.EnvVar {
	serverServiceName := GenerateServerServiceName(instance.Name)
	serverAddress := serverServiceName + "." + instance.Namespace + ".svc.cluster.local"

	envVars := []corev1.EnvVar{
		{
			Name:  "MARIE_CLUSTER_NAME",
			Value: instance.Name,
		},
		{
			Name:  "MARIE_EXECUTOR_GROUP",
			Value: groupSpec.GroupName,
		},
		{
			Name:  "MARIE_EXECUTOR_TYPE",
			Value: groupSpec.ExecutorType,
		},
		{
			Name: "MARIE_POD_NAME",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "metadata.name",
				},
			},
		},
		{
			Name: "MARIE_POD_NAMESPACE",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "metadata.namespace",
				},
			},
		},
		{
			Name: "MARIE_POD_IP",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "status.podIP",
				},
			},
		},
		{
			Name:  "MARIE_NODE_TYPE",
			Value: NodeTypeExecutor,
		},
		{
			Name:  "MARIE_SERVER_ADDRESS",
			Value: serverAddress,
		},
		{
			Name:  "MARIE_SERVER_PORT",
			Value: "52000",
		},
	}

	// Add GPU flag if this is a GPU executor group
	if groupSpec.GPU {
		envVars = append(envVars, corev1.EnvVar{
			Name:  "MARIE_GPU_ENABLED",
			Value: "true",
		})
	}

	return envVars
}

// appendEnvVarsIfNotExist appends environment variables to a list if they don't already exist
func appendEnvVarsIfNotExist(existingVars, newVars []corev1.EnvVar) []corev1.EnvVar {
	existingNames := make(map[string]bool)
	for _, env := range existingVars {
		existingNames[env.Name] = true
	}

	for _, env := range newVars {
		if !existingNames[env.Name] {
			existingVars = append(existingVars, env)
		}
	}

	return existingVars
}

// IsPodRunning checks if a pod is in the Running phase and all containers are ready
func IsPodRunning(pod *corev1.Pod) bool {
	if pod.Status.Phase != corev1.PodRunning {
		return false
	}

	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
			return true
		}
	}

	return false
}

// IsPodReady checks if a pod is ready
func IsPodReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

// GetPodReadyCondition returns the Ready condition for a pod
func GetPodReadyCondition(pod *corev1.Pod) *corev1.PodCondition {
	for i := range pod.Status.Conditions {
		if pod.Status.Conditions[i].Type == corev1.PodReady {
			return &pod.Status.Conditions[i]
		}
	}
	return nil
}

// SetOwnerReference sets the owner reference on an object
func SetOwnerReference(owner metav1.Object, controlled metav1.Object, scheme interface{}) {
	ownerRef := metav1.OwnerReference{
		APIVersion:         "marie.ai/v1alpha1",
		Kind:               "MarieCluster",
		Name:               owner.GetName(),
		UID:                owner.GetUID(),
		Controller:         boolPtr(true),
		BlockOwnerDeletion: boolPtr(true),
	}

	controlledRefs := controlled.GetOwnerReferences()
	for i, ref := range controlledRefs {
		if ref.UID == ownerRef.UID {
			controlledRefs[i] = ownerRef
			controlled.SetOwnerReferences(controlledRefs)
			return
		}
	}
	controlledRefs = append(controlledRefs, ownerRef)
	controlled.SetOwnerReferences(controlledRefs)
}

func boolPtr(b bool) *bool {
	return &b
}
