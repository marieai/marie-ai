package controller

import (
	"context"
	"fmt"
	"reflect"
	"runtime"
	"time"

	"github.com/go-logr/logr"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/manager"

	mariav1alpha1 "github.com/marieai/marie-ai/deploy/operator/api/v1alpha1"
	"github.com/marieai/marie-ai/deploy/operator/internal/controller/common"
)

// reconcileFunc is a function that reconciles a specific aspect of the MarieCluster
type reconcileFunc func(context.Context, *mariav1alpha1.MarieCluster) error

var (
	// DefaultRequeueDuration is the default duration to requeue a reconcile request
	DefaultRequeueDuration = 2 * time.Second
)

// MarieClusterReconciler reconciles a MarieCluster object
type MarieClusterReconciler struct {
	client.Client
	Scheme   *k8sruntime.Scheme
	Recorder record.EventRecorder
	Log      logr.Logger
}

// MarieClusterReconcilerOptions contains options for the reconciler
type MarieClusterReconcilerOptions struct {
	// Add any custom options here
}

// NewMarieClusterReconciler creates a new MarieClusterReconciler
func NewMarieClusterReconciler(mgr manager.Manager) *MarieClusterReconciler {
	return &MarieClusterReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("mariecluster-controller"),
		Log:      ctrl.Log.WithName("controllers").WithName("MarieCluster"),
	}
}

// +kubebuilder:rbac:groups=marie.ai,resources=marieclusters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=marie.ai,resources=marieclusters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=marie.ai,resources=marieclusters/finalizers,verbs=update
// +kubebuilder:rbac:groups=core,resources=events,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;create;update;patch;delete;deletecollection
// +kubebuilder:rbac:groups=core,resources=pods/status,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=networking.k8s.io,resources=ingresses,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=secrets,verbs=get;list;watch

// Reconcile is the main reconciliation loop for MarieCluster
func (r *MarieClusterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := ctrl.LoggerFrom(ctx)

	// Fetch the MarieCluster instance
	instance := &mariav1alpha1.MarieCluster{}
	if err := r.Get(ctx, req.NamespacedName, instance); err != nil {
		if errors.IsNotFound(err) {
			// MarieCluster was deleted, nothing to do
			logger.Info("MarieCluster resource not found, ignoring")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Failed to get MarieCluster")
		return ctrl.Result{}, err
	}

	return r.reconcileMarieCluster(ctx, instance)
}

// reconcileMarieCluster performs the main reconciliation logic
func (r *MarieClusterReconciler) reconcileMarieCluster(ctx context.Context, instance *mariav1alpha1.MarieCluster) (ctrl.Result, error) {
	var reconcileErr error
	logger := ctrl.LoggerFrom(ctx)

	// Handle deletion
	if instance.DeletionTimestamp != nil && !instance.DeletionTimestamp.IsZero() {
		logger.Info("MarieCluster is being deleted")
		return ctrl.Result{}, nil
	}

	// Store original instance for status comparison
	originalInstance := instance.DeepCopy()

	// Chain of reconciliation functions
	reconcileFuncs := []reconcileFunc{
		r.reconcileServerService,
		r.reconcileServerHeadlessService,
		r.reconcileServerDeployment,
		r.reconcileExecutorServices,
		r.reconcileExecutorDeployments,
		r.reconcileIngress,
	}

	// Execute each reconciliation function in order
	for _, fn := range reconcileFuncs {
		if reconcileErr = fn(ctx, instance); reconcileErr != nil {
			funcName := runtime.FuncForPC(reflect.ValueOf(fn).Pointer()).Name()
			logger.Error(reconcileErr, "Error reconciling resources", "function", funcName)
			break
		}
	}

	// Calculate and update status
	newInstance, statusErr := r.calculateStatus(ctx, instance, reconcileErr)
	if statusErr != nil {
		logger.Error(statusErr, "Failed to calculate status")
	}

	// Update status if changed
	var updateErr error
	if newInstance != nil && !reflect.DeepEqual(originalInstance.Status, newInstance.Status) {
		if err := r.Status().Update(ctx, newInstance); err != nil {
			logger.Error(err, "Failed to update MarieCluster status")
			updateErr = err
		} else {
			logger.Info("Updated MarieCluster status", "phase", newInstance.Status.Phase)
		}
	}

	// Determine final error and requeue
	var finalErr error
	if reconcileErr != nil {
		finalErr = reconcileErr
	} else if statusErr != nil {
		finalErr = statusErr
	} else {
		finalErr = updateErr
	}

	if finalErr != nil {
		return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, finalErr
	}

	// Requeue periodically to check status
	return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
}

// reconcileServerService reconciles the main server service
func (r *MarieClusterReconciler) reconcileServerService(ctx context.Context, instance *mariav1alpha1.MarieCluster) error {
	logger := ctrl.LoggerFrom(ctx)
	logger.Info("Reconciling server service")

	service := common.BuildServerService(ctx, instance)

	// Check if service already exists
	existingService := &corev1.Service{}
	err := r.Get(ctx, types.NamespacedName{Name: service.Name, Namespace: service.Namespace}, existingService)
	if err != nil {
		if errors.IsNotFound(err) {
			// Create the service
			if err := ctrl.SetControllerReference(instance, service, r.Scheme); err != nil {
				return err
			}
			if err := r.Create(ctx, service); err != nil {
				r.Recorder.Eventf(instance, corev1.EventTypeWarning, "FailedCreate", "Failed to create server service: %v", err)
				return err
			}
			r.Recorder.Eventf(instance, corev1.EventTypeNormal, "Created", "Created server service %s", service.Name)
			logger.Info("Created server service", "name", service.Name)
			return nil
		}
		return err
	}

	// Service exists, update if needed
	if needsServiceUpdate(existingService, service) {
		existingService.Spec.Ports = service.Spec.Ports
		existingService.Spec.Type = service.Spec.Type
		if err := r.Update(ctx, existingService); err != nil {
			return err
		}
		logger.Info("Updated server service", "name", service.Name)
	}

	return nil
}

// reconcileServerHeadlessService reconciles the headless server service
func (r *MarieClusterReconciler) reconcileServerHeadlessService(ctx context.Context, instance *mariav1alpha1.MarieCluster) error {
	logger := ctrl.LoggerFrom(ctx)
	logger.Info("Reconciling server headless service")

	service := common.BuildServerHeadlessService(ctx, instance)

	// Check if service already exists
	existingService := &corev1.Service{}
	err := r.Get(ctx, types.NamespacedName{Name: service.Name, Namespace: service.Namespace}, existingService)
	if err != nil {
		if errors.IsNotFound(err) {
			// Create the service
			if err := ctrl.SetControllerReference(instance, service, r.Scheme); err != nil {
				return err
			}
			if err := r.Create(ctx, service); err != nil {
				return err
			}
			logger.Info("Created server headless service", "name", service.Name)
			return nil
		}
		return err
	}

	return nil
}

// reconcileServerDeployment reconciles the server deployment
func (r *MarieClusterReconciler) reconcileServerDeployment(ctx context.Context, instance *mariav1alpha1.MarieCluster) error {
	logger := ctrl.LoggerFrom(ctx)
	logger.Info("Reconciling server deployment")

	deployment := common.BuildServerDeployment(ctx, instance)

	// Check if deployment already exists
	existingDeployment := &appsv1.Deployment{}
	err := r.Get(ctx, types.NamespacedName{Name: deployment.Name, Namespace: deployment.Namespace}, existingDeployment)
	if err != nil {
		if errors.IsNotFound(err) {
			// Create the deployment
			if err := ctrl.SetControllerReference(instance, deployment, r.Scheme); err != nil {
				return err
			}
			if err := r.Create(ctx, deployment); err != nil {
				r.Recorder.Eventf(instance, corev1.EventTypeWarning, "FailedCreate", "Failed to create server deployment: %v", err)
				return err
			}
			r.Recorder.Eventf(instance, corev1.EventTypeNormal, "Created", "Created server deployment %s", deployment.Name)
			logger.Info("Created server deployment", "name", deployment.Name)
			return nil
		}
		return err
	}

	// Deployment exists, update if needed
	if needsDeploymentUpdate(existingDeployment, deployment) {
		existingDeployment.Spec.Replicas = deployment.Spec.Replicas
		existingDeployment.Spec.Template = deployment.Spec.Template
		if err := r.Update(ctx, existingDeployment); err != nil {
			return err
		}
		logger.Info("Updated server deployment", "name", deployment.Name)
	}

	return nil
}

// reconcileExecutorServices reconciles services for all executor groups
func (r *MarieClusterReconciler) reconcileExecutorServices(ctx context.Context, instance *mariav1alpha1.MarieCluster) error {
	logger := ctrl.LoggerFrom(ctx)

	for _, groupSpec := range instance.Spec.ExecutorGroupSpecs {
		logger.Info("Reconciling executor service", "group", groupSpec.GroupName)

		service := common.BuildExecutorService(ctx, instance, &groupSpec)

		// Check if service already exists
		existingService := &corev1.Service{}
		err := r.Get(ctx, types.NamespacedName{Name: service.Name, Namespace: service.Namespace}, existingService)
		if err != nil {
			if errors.IsNotFound(err) {
				// Create the service
				if err := ctrl.SetControllerReference(instance, service, r.Scheme); err != nil {
					return err
				}
				if err := r.Create(ctx, service); err != nil {
					return err
				}
				logger.Info("Created executor service", "name", service.Name, "group", groupSpec.GroupName)
				continue
			}
			return err
		}
	}

	return nil
}

// reconcileExecutorDeployments reconciles deployments for all executor groups
func (r *MarieClusterReconciler) reconcileExecutorDeployments(ctx context.Context, instance *mariav1alpha1.MarieCluster) error {
	logger := ctrl.LoggerFrom(ctx)

	for _, groupSpec := range instance.Spec.ExecutorGroupSpecs {
		logger.Info("Reconciling executor deployment", "group", groupSpec.GroupName)

		deployment := common.BuildExecutorDeployment(ctx, instance, &groupSpec)

		// Check if deployment already exists
		existingDeployment := &appsv1.Deployment{}
		err := r.Get(ctx, types.NamespacedName{Name: deployment.Name, Namespace: deployment.Namespace}, existingDeployment)
		if err != nil {
			if errors.IsNotFound(err) {
				// Create the deployment
				if err := ctrl.SetControllerReference(instance, deployment, r.Scheme); err != nil {
					return err
				}
				if err := r.Create(ctx, deployment); err != nil {
					r.Recorder.Eventf(instance, corev1.EventTypeWarning, "FailedCreate", "Failed to create executor deployment: %v", err)
					return err
				}
				r.Recorder.Eventf(instance, corev1.EventTypeNormal, "Created", "Created executor deployment %s", deployment.Name)
				logger.Info("Created executor deployment", "name", deployment.Name, "group", groupSpec.GroupName)
				continue
			}
			return err
		}

		// Deployment exists, update if needed
		if needsDeploymentUpdate(existingDeployment, deployment) {
			existingDeployment.Spec.Replicas = deployment.Spec.Replicas
			existingDeployment.Spec.Template = deployment.Spec.Template
			if err := r.Update(ctx, existingDeployment); err != nil {
				return err
			}
			logger.Info("Updated executor deployment", "name", deployment.Name, "group", groupSpec.GroupName)
		}
	}

	// Clean up any orphaned executor deployments
	return r.cleanupOrphanedExecutorDeployments(ctx, instance)
}

// cleanupOrphanedExecutorDeployments removes deployments for executor groups that no longer exist
func (r *MarieClusterReconciler) cleanupOrphanedExecutorDeployments(ctx context.Context, instance *mariav1alpha1.MarieCluster) error {
	logger := ctrl.LoggerFrom(ctx)

	// List all executor deployments for this cluster
	deploymentList := &appsv1.DeploymentList{}
	if err := r.List(ctx, deploymentList, client.InNamespace(instance.Namespace), client.MatchingLabels{
		common.MarieClusterLabelKey:  instance.Name,
		common.MarieNodeTypeLabelKey: common.NodeTypeExecutor,
	}); err != nil {
		return err
	}

	// Build a set of expected executor group names
	expectedGroups := make(map[string]bool)
	for _, groupSpec := range instance.Spec.ExecutorGroupSpecs {
		expectedGroups[groupSpec.GroupName] = true
	}

	// Delete deployments for groups that no longer exist
	for _, deployment := range deploymentList.Items {
		groupName, ok := deployment.Labels[common.MarieNodeGroupLabelKey]
		if !ok {
			continue
		}

		if !expectedGroups[groupName] {
			logger.Info("Deleting orphaned executor deployment", "name", deployment.Name, "group", groupName)
			if err := r.Delete(ctx, &deployment); err != nil && !errors.IsNotFound(err) {
				return err
			}
			r.Recorder.Eventf(instance, corev1.EventTypeNormal, "Deleted", "Deleted orphaned executor deployment %s", deployment.Name)
		}
	}

	return nil
}

// reconcileIngress reconciles the ingress if enabled
func (r *MarieClusterReconciler) reconcileIngress(ctx context.Context, instance *mariav1alpha1.MarieCluster) error {
	logger := ctrl.LoggerFrom(ctx)

	if instance.Spec.EnableIngress == nil || !*instance.Spec.EnableIngress {
		// Ingress not enabled, clean up if exists
		return r.cleanupIngress(ctx, instance)
	}

	logger.Info("Reconciling ingress")

	ingress := r.buildIngress(instance)

	// Check if ingress already exists
	existingIngress := &networkingv1.Ingress{}
	err := r.Get(ctx, types.NamespacedName{Name: ingress.Name, Namespace: ingress.Namespace}, existingIngress)
	if err != nil {
		if errors.IsNotFound(err) {
			// Create the ingress
			if err := ctrl.SetControllerReference(instance, ingress, r.Scheme); err != nil {
				return err
			}
			if err := r.Create(ctx, ingress); err != nil {
				return err
			}
			logger.Info("Created ingress", "name", ingress.Name)
			return nil
		}
		return err
	}

	return nil
}

// cleanupIngress removes the ingress if it exists
func (r *MarieClusterReconciler) cleanupIngress(ctx context.Context, instance *mariav1alpha1.MarieCluster) error {
	ingress := &networkingv1.Ingress{}
	err := r.Get(ctx, types.NamespacedName{Name: instance.Name + "-ingress", Namespace: instance.Namespace}, ingress)
	if err != nil {
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}

	return r.Delete(ctx, ingress)
}

// buildIngress builds an Ingress for the MarieCluster
func (r *MarieClusterReconciler) buildIngress(instance *mariav1alpha1.MarieCluster) *networkingv1.Ingress {
	pathType := networkingv1.PathTypePrefix

	return &networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      instance.Name + "-ingress",
			Namespace: instance.Namespace,
			Labels:    common.ClusterLabels(instance.Name),
		},
		Spec: networkingv1.IngressSpec{
			Rules: []networkingv1.IngressRule{
				{
					IngressRuleValue: networkingv1.IngressRuleValue{
						HTTP: &networkingv1.HTTPIngressRuleValue{
							Paths: []networkingv1.HTTPIngressPath{
								{
									Path:     "/",
									PathType: &pathType,
									Backend: networkingv1.IngressBackend{
										Service: &networkingv1.IngressServiceBackend{
											Name: common.GenerateServerServiceName(instance.Name),
											Port: networkingv1.ServiceBackendPort{
												Number: common.DefaultHTTPPort,
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
}

// calculateStatus calculates the status of the MarieCluster
func (r *MarieClusterReconciler) calculateStatus(ctx context.Context, instance *mariav1alpha1.MarieCluster, reconcileErr error) (*mariav1alpha1.MarieCluster, error) {
	logger := ctrl.LoggerFrom(ctx)

	newInstance := instance.DeepCopy()
	now := metav1.Now()
	newInstance.Status.LastUpdateTime = &now
	newInstance.Status.ObservedGeneration = instance.Generation

	// Initialize executor group statuses
	if newInstance.Status.ExecutorGroupStatuses == nil {
		newInstance.Status.ExecutorGroupStatuses = make(map[string]mariav1alpha1.ExecutorGroupStatus)
	}

	// Check server deployment status
	serverDeployment := &appsv1.Deployment{}
	serverDeploymentName := common.GenerateServerDeploymentName(instance.Name)
	if err := r.Get(ctx, types.NamespacedName{Name: serverDeploymentName, Namespace: instance.Namespace}, serverDeployment); err != nil {
		if !errors.IsNotFound(err) {
			return nil, err
		}
		newInstance.Status.ServerReady = false
	} else {
		newInstance.Status.ServerReady = serverDeployment.Status.ReadyReplicas > 0

		// Get server pod info
		podList := &corev1.PodList{}
		if err := r.List(ctx, podList, common.ClusterServerPodsAssociationOptions(instance).ToListOptions()...); err == nil {
			for _, pod := range podList.Items {
				if common.IsPodReady(&pod) {
					newInstance.Status.ServerInfo.PodName = pod.Name
					newInstance.Status.ServerInfo.PodIP = pod.Status.PodIP
					break
				}
			}
		}
		newInstance.Status.ServerInfo.ServiceName = common.GenerateServerServiceName(instance.Name)
	}

	// Check executor deployments status
	var totalDesired, totalReady, totalAvailable int32
	var totalCPU, totalMemory, totalGPU resource.Quantity

	for _, groupSpec := range instance.Spec.ExecutorGroupSpecs {
		deployment := &appsv1.Deployment{}
		deploymentName := common.GenerateExecutorDeploymentName(instance.Name, groupSpec.GroupName)
		if err := r.Get(ctx, types.NamespacedName{Name: deploymentName, Namespace: instance.Namespace}, deployment); err != nil {
			if !errors.IsNotFound(err) {
				logger.Error(err, "Failed to get executor deployment", "group", groupSpec.GroupName)
			}
			continue
		}

		groupStatus := mariav1alpha1.ExecutorGroupStatus{
			Replicas:          deployment.Status.Replicas,
			ReadyReplicas:     deployment.Status.ReadyReplicas,
			AvailableReplicas: deployment.Status.AvailableReplicas,
		}
		newInstance.Status.ExecutorGroupStatuses[groupSpec.GroupName] = groupStatus

		if groupSpec.Replicas != nil {
			totalDesired += *groupSpec.Replicas
		}
		totalReady += deployment.Status.ReadyReplicas
		totalAvailable += deployment.Status.AvailableReplicas

		// Calculate resource totals
		if len(groupSpec.Template.Spec.Containers) > 0 {
			container := groupSpec.Template.Spec.Containers[0]
			replicas := int32(1)
			if groupSpec.Replicas != nil {
				replicas = *groupSpec.Replicas
			}

			if cpu := container.Resources.Requests.Cpu(); cpu != nil {
				cpuTotal := cpu.DeepCopy()
				for i := int32(0); i < replicas; i++ {
					totalCPU.Add(cpuTotal)
				}
			}
			if mem := container.Resources.Requests.Memory(); mem != nil {
				memTotal := mem.DeepCopy()
				for i := int32(0); i < replicas; i++ {
					totalMemory.Add(memTotal)
				}
			}
			if gpu := container.Resources.Limits["nvidia.com/gpu"]; !gpu.IsZero() {
				gpuTotal := gpu.DeepCopy()
				for i := int32(0); i < replicas; i++ {
					totalGPU.Add(gpuTotal)
				}
			}
		}
	}

	newInstance.Status.DesiredExecutorReplicas = totalDesired
	newInstance.Status.ReadyExecutorReplicas = totalReady
	newInstance.Status.AvailableExecutorReplicas = totalAvailable
	newInstance.Status.DesiredCPU = totalCPU
	newInstance.Status.DesiredMemory = totalMemory
	newInstance.Status.DesiredGPU = totalGPU

	// Set endpoints
	newInstance.Status.Endpoints = map[string]string{
		"grpc": fmt.Sprintf("%s.%s.svc.cluster.local:%d",
			common.GenerateServerServiceName(instance.Name), instance.Namespace, common.DefaultServerPort),
		"http": fmt.Sprintf("%s.%s.svc.cluster.local:%d",
			common.GenerateServerServiceName(instance.Name), instance.Namespace, common.DefaultHTTPPort),
	}

	// Determine phase
	newInstance.Status.Phase = r.determinePhase(newInstance, reconcileErr)

	// Set conditions
	r.setConditions(newInstance, reconcileErr)

	return newInstance, nil
}

// determinePhase determines the current phase of the cluster
func (r *MarieClusterReconciler) determinePhase(instance *mariav1alpha1.MarieCluster, reconcileErr error) mariav1alpha1.ClusterPhase {
	// Check if suspended
	if instance.Spec.Suspend != nil && *instance.Spec.Suspend {
		return mariav1alpha1.ClusterPhaseSuspended
	}

	// Check for errors
	if reconcileErr != nil {
		return mariav1alpha1.ClusterPhaseFailed
	}

	// Check if server is ready
	if !instance.Status.ServerReady {
		if instance.Status.Phase == "" {
			return mariav1alpha1.ClusterPhasePending
		}
		return mariav1alpha1.ClusterPhaseCreating
	}

	// Check if all executors are ready
	if instance.Status.ReadyExecutorReplicas < instance.Status.DesiredExecutorReplicas {
		return mariav1alpha1.ClusterPhaseCreating
	}

	return mariav1alpha1.ClusterPhaseRunning
}

// setConditions sets the status conditions for the cluster
func (r *MarieClusterReconciler) setConditions(instance *mariav1alpha1.MarieCluster, reconcileErr error) {
	now := metav1.Now()

	// MarieClusterProvisioned condition
	provisionedCondition := metav1.Condition{
		Type:               mariav1alpha1.MarieClusterProvisioned,
		LastTransitionTime: now,
	}
	if reconcileErr != nil {
		provisionedCondition.Status = metav1.ConditionFalse
		provisionedCondition.Reason = "ProvisioningFailed"
		provisionedCondition.Message = reconcileErr.Error()
	} else {
		provisionedCondition.Status = metav1.ConditionTrue
		provisionedCondition.Reason = "Provisioned"
		provisionedCondition.Message = "Cluster resources have been created"
	}
	meta.SetStatusCondition(&instance.Status.Conditions, provisionedCondition)

	// ServerPodReady condition
	serverReadyCondition := metav1.Condition{
		Type:               mariav1alpha1.ServerPodReady,
		LastTransitionTime: now,
	}
	if instance.Status.ServerReady {
		serverReadyCondition.Status = metav1.ConditionTrue
		serverReadyCondition.Reason = "ServerReady"
		serverReadyCondition.Message = "Server pod is ready"
	} else {
		serverReadyCondition.Status = metav1.ConditionFalse
		serverReadyCondition.Reason = "ServerNotReady"
		serverReadyCondition.Message = "Server pod is not ready"
	}
	meta.SetStatusCondition(&instance.Status.Conditions, serverReadyCondition)

	// AllExecutorsReady condition
	executorsReadyCondition := metav1.Condition{
		Type:               mariav1alpha1.AllExecutorsReady,
		LastTransitionTime: now,
	}
	if instance.Status.ReadyExecutorReplicas >= instance.Status.DesiredExecutorReplicas {
		executorsReadyCondition.Status = metav1.ConditionTrue
		executorsReadyCondition.Reason = "AllExecutorsReady"
		executorsReadyCondition.Message = fmt.Sprintf("%d/%d executor replicas are ready",
			instance.Status.ReadyExecutorReplicas, instance.Status.DesiredExecutorReplicas)
	} else {
		executorsReadyCondition.Status = metav1.ConditionFalse
		executorsReadyCondition.Reason = "ExecutorsNotReady"
		executorsReadyCondition.Message = fmt.Sprintf("%d/%d executor replicas are ready",
			instance.Status.ReadyExecutorReplicas, instance.Status.DesiredExecutorReplicas)
	}
	meta.SetStatusCondition(&instance.Status.Conditions, executorsReadyCondition)
}

// needsServiceUpdate checks if a service needs to be updated
func needsServiceUpdate(existing, desired *corev1.Service) bool {
	if existing.Spec.Type != desired.Spec.Type {
		return true
	}
	// Compare ports
	if len(existing.Spec.Ports) != len(desired.Spec.Ports) {
		return true
	}
	return false
}

// needsDeploymentUpdate checks if a deployment needs to be updated
func needsDeploymentUpdate(existing, desired *appsv1.Deployment) bool {
	if existing.Spec.Replicas == nil || desired.Spec.Replicas == nil {
		return existing.Spec.Replicas != desired.Spec.Replicas
	}
	if *existing.Spec.Replicas != *desired.Spec.Replicas {
		return true
	}
	// Could add more sophisticated comparison here
	return false
}

// SetupWithManager sets up the controller with the Manager
func (r *MarieClusterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&mariav1alpha1.MarieCluster{}).
		Owns(&appsv1.Deployment{}).
		Owns(&corev1.Service{}).
		Owns(&networkingv1.Ingress{}).
		Complete(r)
}
