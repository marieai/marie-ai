package common

import (
	"context"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	ctrl "sigs.k8s.io/controller-runtime"

	mariav1alpha1 "github.com/marieai/marie-ai/deploy/operator/api/v1alpha1"
)

const (
	// DefaultServerPort is the default gRPC port for Marie server
	DefaultServerPort = 52000
	// DefaultHTTPPort is the default HTTP port for Marie server
	DefaultHTTPPort = 8080
	// DefaultMetricsPort is the default metrics port
	DefaultMetricsPort = 9090
	// DefaultExecutorPort is the default executor gRPC port
	DefaultExecutorPort = 52001
)

// BuildServerService builds the main service for the Marie server
func BuildServerService(ctx context.Context, instance *mariav1alpha1.MarieCluster) *corev1.Service {
	log := ctrl.LoggerFrom(ctx)
	log.Info("Building server service", "cluster", instance.Name)

	labels := ServerLabels(instance.Name)
	selector := ServerLabels(instance.Name)

	// Determine service type
	serviceType := instance.Spec.ServiceType
	if serviceType == "" {
		serviceType = corev1.ServiceTypeClusterIP
	}

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      GenerateServerServiceName(instance.Name),
			Namespace: instance.Namespace,
			Labels:    labels,
		},
		Spec: corev1.ServiceSpec{
			Type:     serviceType,
			Selector: selector,
			Ports: []corev1.ServicePort{
				{
					Name:       "grpc",
					Port:       DefaultServerPort,
					TargetPort: intstr.FromInt(DefaultServerPort),
					Protocol:   corev1.ProtocolTCP,
				},
				{
					Name:       "http",
					Port:       DefaultHTTPPort,
					TargetPort: intstr.FromInt(DefaultHTTPPort),
					Protocol:   corev1.ProtocolTCP,
				},
				{
					Name:       "metrics",
					Port:       DefaultMetricsPort,
					TargetPort: intstr.FromInt(DefaultMetricsPort),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}

	return service
}

// BuildServerHeadlessService builds a headless service for the Marie server
// This is useful for StatefulSet-style deployments and internal DNS resolution
func BuildServerHeadlessService(ctx context.Context, instance *mariav1alpha1.MarieCluster) *corev1.Service {
	log := ctrl.LoggerFrom(ctx)
	log.Info("Building server headless service", "cluster", instance.Name)

	labels := ServerLabels(instance.Name)
	selector := ServerLabels(instance.Name)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      GenerateServerHeadlessServiceName(instance.Name),
			Namespace: instance.Namespace,
			Labels:    labels,
		},
		Spec: corev1.ServiceSpec{
			Type:                     corev1.ServiceTypeClusterIP,
			ClusterIP:                corev1.ClusterIPNone,
			Selector:                 selector,
			PublishNotReadyAddresses: true,
			Ports: []corev1.ServicePort{
				{
					Name:       "grpc",
					Port:       DefaultServerPort,
					TargetPort: intstr.FromInt(DefaultServerPort),
					Protocol:   corev1.ProtocolTCP,
				},
				{
					Name:       "http",
					Port:       DefaultHTTPPort,
					TargetPort: intstr.FromInt(DefaultHTTPPort),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}

	return service
}

// BuildExecutorService builds a service for an executor group
func BuildExecutorService(ctx context.Context, instance *mariav1alpha1.MarieCluster, groupSpec *mariav1alpha1.ExecutorGroupSpec) *corev1.Service {
	log := ctrl.LoggerFrom(ctx)
	log.Info("Building executor service", "cluster", instance.Name, "group", groupSpec.GroupName)

	labels := ExecutorLabels(instance.Name, groupSpec.GroupName)
	selector := ExecutorLabels(instance.Name, groupSpec.GroupName)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      GenerateExecutorServiceName(instance.Name, groupSpec.GroupName),
			Namespace: instance.Namespace,
			Labels:    labels,
		},
		Spec: corev1.ServiceSpec{
			Type:      corev1.ServiceTypeClusterIP,
			ClusterIP: corev1.ClusterIPNone, // Headless service for executors
			Selector:  selector,
			Ports: []corev1.ServicePort{
				{
					Name:       "grpc",
					Port:       DefaultExecutorPort,
					TargetPort: intstr.FromInt(DefaultExecutorPort),
					Protocol:   corev1.ProtocolTCP,
				},
				{
					Name:       "metrics",
					Port:       DefaultMetricsPort,
					TargetPort: intstr.FromInt(DefaultMetricsPort),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}

	return service
}
