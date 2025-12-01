package controller

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/go-logr/logr"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
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

// MarieJobReconciler reconciles a MarieJob object
type MarieJobReconciler struct {
	client.Client
	Scheme     *k8sruntime.Scheme
	Recorder   record.EventRecorder
	Log        logr.Logger
	HTTPClient *http.Client
}

// NewMarieJobReconciler creates a new MarieJobReconciler
func NewMarieJobReconciler(mgr manager.Manager) *MarieJobReconciler {
	return &MarieJobReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("mariejob-controller"),
		Log:      ctrl.Log.WithName("controllers").WithName("MarieJob"),
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// +kubebuilder:rbac:groups=marie.ai,resources=mariejobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=marie.ai,resources=mariejobs/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=marie.ai,resources=mariejobs/finalizers,verbs=update
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=secrets,verbs=get;list;watch

// Reconcile is the main reconciliation loop for MarieJob
func (r *MarieJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := ctrl.LoggerFrom(ctx)

	// Fetch the MarieJob instance
	instance := &mariav1alpha1.MarieJob{}
	if err := r.Get(ctx, req.NamespacedName, instance); err != nil {
		if errors.IsNotFound(err) {
			logger.Info("MarieJob resource not found, ignoring")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Failed to get MarieJob")
		return ctrl.Result{}, err
	}

	return r.reconcileMarieJob(ctx, instance)
}

// reconcileMarieJob performs the main reconciliation logic
func (r *MarieJobReconciler) reconcileMarieJob(ctx context.Context, instance *mariav1alpha1.MarieJob) (ctrl.Result, error) {
	logger := ctrl.LoggerFrom(ctx)

	// Handle deletion
	if instance.DeletionTimestamp != nil && !instance.DeletionTimestamp.IsZero() {
		logger.Info("MarieJob is being deleted")
		return ctrl.Result{}, nil
	}

	// Check if job is already completed or failed
	if instance.Status.Phase == mariav1alpha1.JobPhaseSucceeded ||
		instance.Status.Phase == mariav1alpha1.JobPhaseFailed ||
		instance.Status.Phase == mariav1alpha1.JobPhaseCancelled {
		// Check TTL for cleanup
		return r.handleCompletedJob(ctx, instance)
	}

	// Check if job is suspended
	if instance.Spec.Suspend {
		return r.handleSuspendedJob(ctx, instance)
	}

	// Find the target cluster
	cluster, err := r.findTargetCluster(ctx, instance)
	if err != nil {
		logger.Error(err, "Failed to find target cluster")
		return r.updateJobStatus(ctx, instance, mariav1alpha1.JobPhasePending, "Waiting for cluster", err)
	}

	// Check if cluster is ready
	if cluster.Status.Phase != mariav1alpha1.ClusterPhaseRunning {
		logger.Info("Cluster is not running, waiting", "cluster", cluster.Name, "phase", cluster.Status.Phase)
		return r.updateJobStatus(ctx, instance, mariav1alpha1.JobPhasePending, "Waiting for cluster to be ready", nil)
	}

	// Submit the job based on submission mode
	switch instance.Spec.SubmissionMode {
	case mariav1alpha1.HTTPSubmission:
		return r.handleHTTPSubmission(ctx, instance, cluster)
	case mariav1alpha1.K8sJobSubmission:
		return r.handleK8sJobSubmission(ctx, instance, cluster)
	default:
		// Default to HTTP submission
		return r.handleHTTPSubmission(ctx, instance, cluster)
	}
}

// findTargetCluster finds the target cluster for the job
func (r *MarieJobReconciler) findTargetCluster(ctx context.Context, instance *mariav1alpha1.MarieJob) (*mariav1alpha1.MarieCluster, error) {
	// Try ClusterRef first
	if instance.Spec.ClusterRef != "" {
		cluster := &mariav1alpha1.MarieCluster{}
		if err := r.Get(ctx, types.NamespacedName{
			Name:      instance.Spec.ClusterRef,
			Namespace: instance.Namespace,
		}, cluster); err != nil {
			return nil, err
		}
		return cluster, nil
	}

	// Try ClusterSelector
	if len(instance.Spec.ClusterSelector) > 0 {
		clusterList := &mariav1alpha1.MarieClusterList{}
		if err := r.List(ctx, clusterList,
			client.InNamespace(instance.Namespace),
			client.MatchingLabels(instance.Spec.ClusterSelector)); err != nil {
			return nil, err
		}

		if len(clusterList.Items) == 0 {
			return nil, fmt.Errorf("no cluster found matching selector")
		}

		// Return the first running cluster
		for _, cluster := range clusterList.Items {
			if cluster.Status.Phase == mariav1alpha1.ClusterPhaseRunning {
				return &cluster, nil
			}
		}

		// Return the first cluster if none are running
		return &clusterList.Items[0], nil
	}

	// Look for any running cluster in the namespace
	clusterList := &mariav1alpha1.MarieClusterList{}
	if err := r.List(ctx, clusterList, client.InNamespace(instance.Namespace)); err != nil {
		return nil, err
	}

	if len(clusterList.Items) == 0 {
		return nil, fmt.Errorf("no cluster found in namespace %s", instance.Namespace)
	}

	// Return the first running cluster
	for _, cluster := range clusterList.Items {
		if cluster.Status.Phase == mariav1alpha1.ClusterPhaseRunning {
			return &cluster, nil
		}
	}

	return &clusterList.Items[0], nil
}

// handleHTTPSubmission handles job submission via HTTP API
func (r *MarieJobReconciler) handleHTTPSubmission(ctx context.Context, instance *mariav1alpha1.MarieJob, cluster *mariav1alpha1.MarieCluster) (ctrl.Result, error) {
	logger := ctrl.LoggerFrom(ctx)

	// Check if job is already submitted
	if instance.Status.JobID != "" {
		// Poll for job status
		return r.pollJobStatus(ctx, instance, cluster)
	}

	// Submit the job
	logger.Info("Submitting job via HTTP", "job", instance.Name, "cluster", cluster.Name)

	// Build the request payload
	payload := r.buildJobPayload(instance)

	// Get the cluster endpoint
	serverService := common.GenerateServerServiceName(cluster.Name)
	endpoint := fmt.Sprintf("http://%s.%s.svc.cluster.local:%d/api/v1/jobs",
		serverService, cluster.Namespace, common.DefaultHTTPPort)

	// Submit the job
	jobID, err := r.submitJobHTTP(ctx, endpoint, payload)
	if err != nil {
		logger.Error(err, "Failed to submit job via HTTP")
		return r.updateJobStatus(ctx, instance, mariav1alpha1.JobPhaseInitializing, "Failed to submit job", err)
	}

	// Update status with job ID
	instance.Status.JobID = jobID
	instance.Status.ClusterName = cluster.Name
	now := metav1.Now()
	instance.Status.StartTime = &now

	return r.updateJobStatus(ctx, instance, mariav1alpha1.JobPhaseRunning, "Job submitted", nil)
}

// submitJobHTTP submits a job via HTTP and returns the job ID
func (r *MarieJobReconciler) submitJobHTTP(ctx context.Context, endpoint string, payload map[string]interface{}) (string, error) {
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.HTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to submit job: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("job submission failed with status %d: %s", resp.StatusCode, string(body))
	}

	var response struct {
		JobID string `json:"job_id"`
		DagID string `json:"dag_id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	return response.JobID, nil
}

// pollJobStatus polls the job status from the Marie server
func (r *MarieJobReconciler) pollJobStatus(ctx context.Context, instance *mariav1alpha1.MarieJob, cluster *mariav1alpha1.MarieCluster) (ctrl.Result, error) {
	logger := ctrl.LoggerFrom(ctx)

	serverService := common.GenerateServerServiceName(cluster.Name)
	endpoint := fmt.Sprintf("http://%s.%s.svc.cluster.local:%d/api/v1/jobs/%s",
		serverService, cluster.Namespace, common.DefaultHTTPPort, instance.Status.JobID)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		logger.Error(err, "Failed to create status request")
		return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
	}

	resp, err := r.HTTPClient.Do(req)
	if err != nil {
		logger.Error(err, "Failed to get job status")
		return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		logger.Info("Job status check returned non-OK status", "status", resp.StatusCode)
		return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
	}

	var status struct {
		Status   string `json:"status"`
		Progress int32  `json:"progress"`
		Error    string `json:"error"`
		Result   struct {
			OutputURI     string            `json:"output_uri"`
			DocumentCount int32             `json:"document_count"`
			PageCount     int32             `json:"page_count"`
			Summary       map[string]string `json:"summary"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		logger.Error(err, "Failed to decode job status")
		return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
	}

	// Update progress
	instance.Status.Progress = status.Progress

	// Map status to phase
	switch status.Status {
	case "completed", "succeeded":
		now := metav1.Now()
		instance.Status.CompletionTime = &now
		if instance.Status.StartTime != nil {
			instance.Status.Duration = now.Sub(instance.Status.StartTime.Time).String()
		}
		instance.Status.Result = &mariav1alpha1.JobResult{
			OutputURI:     status.Result.OutputURI,
			DocumentCount: status.Result.DocumentCount,
			PageCount:     status.Result.PageCount,
			Summary:       status.Result.Summary,
		}
		return r.updateJobStatus(ctx, instance, mariav1alpha1.JobPhaseSucceeded, "Job completed", nil)

	case "failed":
		now := metav1.Now()
		instance.Status.CompletionTime = &now
		if instance.Status.StartTime != nil {
			instance.Status.Duration = now.Sub(instance.Status.StartTime.Time).String()
		}
		instance.Status.Error = &mariav1alpha1.JobError{
			Message: status.Error,
		}
		return r.updateJobStatus(ctx, instance, mariav1alpha1.JobPhaseFailed, "Job failed", nil)

	case "cancelled":
		return r.updateJobStatus(ctx, instance, mariav1alpha1.JobPhaseCancelled, "Job cancelled", nil)

	default:
		// Still running
		return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
	}
}

// handleK8sJobSubmission handles job submission via Kubernetes Job
func (r *MarieJobReconciler) handleK8sJobSubmission(ctx context.Context, instance *mariav1alpha1.MarieJob, cluster *mariav1alpha1.MarieCluster) (ctrl.Result, error) {
	logger := ctrl.LoggerFrom(ctx)

	// Check if submitter job already exists
	if instance.Status.SubmitterJobName != "" {
		return r.monitorSubmitterJob(ctx, instance)
	}

	logger.Info("Creating submitter job", "job", instance.Name, "cluster", cluster.Name)

	// Create the submitter job
	submitterJob := r.buildSubmitterJob(instance, cluster)

	if err := ctrl.SetControllerReference(instance, submitterJob, r.Scheme); err != nil {
		return ctrl.Result{}, err
	}

	if err := r.Create(ctx, submitterJob); err != nil {
		if errors.IsAlreadyExists(err) {
			// Job already exists, monitor it
			instance.Status.SubmitterJobName = submitterJob.Name
			return r.monitorSubmitterJob(ctx, instance)
		}
		return r.updateJobStatus(ctx, instance, mariav1alpha1.JobPhaseInitializing, "Failed to create submitter job", err)
	}

	instance.Status.SubmitterJobName = submitterJob.Name
	instance.Status.ClusterName = cluster.Name
	now := metav1.Now()
	instance.Status.StartTime = &now

	r.Recorder.Eventf(instance, corev1.EventTypeNormal, "Created", "Created submitter job %s", submitterJob.Name)

	return r.updateJobStatus(ctx, instance, mariav1alpha1.JobPhaseRunning, "Submitter job created", nil)
}

// buildSubmitterJob builds the Kubernetes Job for job submission
func (r *MarieJobReconciler) buildSubmitterJob(instance *mariav1alpha1.MarieJob, cluster *mariav1alpha1.MarieCluster) *batchv1.Job {
	labels := common.JobLabels(instance.Name, cluster.Name)

	// Use custom pod template if provided
	var podSpec corev1.PodSpec
	if instance.Spec.SubmitterPodTemplate != nil {
		podSpec = instance.Spec.SubmitterPodTemplate.Spec
	} else {
		// Build default pod spec
		serverEndpoint := fmt.Sprintf("http://%s.%s.svc.cluster.local:%d",
			common.GenerateServerServiceName(cluster.Name), cluster.Namespace, common.DefaultHTTPPort)

		podSpec = corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{
				{
					Name:  "submitter",
					Image: "curlimages/curl:latest",
					Command: []string{
						"/bin/sh",
						"-c",
						fmt.Sprintf(`
							curl -X POST %s/api/v1/jobs \
								-H "Content-Type: application/json" \
								-d '{
									"job_type": "%s",
									"input": "%s",
									"options": {}
								}'
						`, serverEndpoint, instance.Spec.JobType, getInputURI(instance)),
					},
				},
			},
		}
	}

	backoffLimit := int32(3)
	if instance.Spec.BackoffLimit != nil {
		backoffLimit = *instance.Spec.BackoffLimit
	}

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      instance.Name + "-submitter",
			Namespace: instance.Namespace,
			Labels:    labels,
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoffLimit,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: podSpec,
			},
		},
	}
}

// monitorSubmitterJob monitors the submitter Kubernetes Job
func (r *MarieJobReconciler) monitorSubmitterJob(ctx context.Context, instance *mariav1alpha1.MarieJob) (ctrl.Result, error) {
	logger := ctrl.LoggerFrom(ctx)

	job := &batchv1.Job{}
	if err := r.Get(ctx, types.NamespacedName{
		Name:      instance.Status.SubmitterJobName,
		Namespace: instance.Namespace,
	}, job); err != nil {
		if errors.IsNotFound(err) {
			logger.Info("Submitter job not found")
			return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
		}
		return ctrl.Result{}, err
	}

	// Check job status
	if job.Status.Succeeded > 0 {
		now := metav1.Now()
		instance.Status.CompletionTime = &now
		if instance.Status.StartTime != nil {
			instance.Status.Duration = now.Sub(instance.Status.StartTime.Time).String()
		}
		return r.updateJobStatus(ctx, instance, mariav1alpha1.JobPhaseSucceeded, "Job completed", nil)
	}

	if job.Status.Failed > 0 {
		now := metav1.Now()
		instance.Status.CompletionTime = &now
		if instance.Status.StartTime != nil {
			instance.Status.Duration = now.Sub(instance.Status.StartTime.Time).String()
		}
		return r.updateJobStatus(ctx, instance, mariav1alpha1.JobPhaseFailed, "Submitter job failed", nil)
	}

	// Still running
	return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
}

// handleCompletedJob handles cleanup for completed jobs
func (r *MarieJobReconciler) handleCompletedJob(ctx context.Context, instance *mariav1alpha1.MarieJob) (ctrl.Result, error) {
	if instance.Spec.TTLSecondsAfterFinished == nil {
		return ctrl.Result{}, nil
	}

	if instance.Status.CompletionTime == nil {
		return ctrl.Result{}, nil
	}

	ttl := time.Duration(*instance.Spec.TTLSecondsAfterFinished) * time.Second
	elapsed := time.Since(instance.Status.CompletionTime.Time)

	if elapsed >= ttl {
		// Delete the job
		if err := r.Delete(ctx, instance); err != nil && !errors.IsNotFound(err) {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}

	// Requeue when TTL expires
	return ctrl.Result{RequeueAfter: ttl - elapsed}, nil
}

// handleSuspendedJob handles suspended jobs
func (r *MarieJobReconciler) handleSuspendedJob(ctx context.Context, instance *mariav1alpha1.MarieJob) (ctrl.Result, error) {
	if instance.Status.Phase != mariav1alpha1.JobPhasePending {
		return r.updateJobStatus(ctx, instance, mariav1alpha1.JobPhasePending, "Job suspended", nil)
	}
	return ctrl.Result{}, nil
}

// updateJobStatus updates the job status
func (r *MarieJobReconciler) updateJobStatus(ctx context.Context, instance *mariav1alpha1.MarieJob, phase mariav1alpha1.JobPhase, message string, reconcileErr error) (ctrl.Result, error) {
	instance.Status.Phase = phase
	instance.Status.Message = message
	instance.Status.ObservedGeneration = instance.Generation

	// Set conditions
	now := metav1.Now()
	condition := metav1.Condition{
		Type:               string(phase),
		Status:             metav1.ConditionTrue,
		LastTransitionTime: now,
		Reason:             string(phase),
		Message:            message,
	}
	if reconcileErr != nil {
		condition.Message = reconcileErr.Error()
	}
	meta.SetStatusCondition(&instance.Status.Conditions, condition)

	if err := r.Status().Update(ctx, instance); err != nil {
		return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
	}

	if reconcileErr != nil {
		return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, reconcileErr
	}

	// Determine requeue based on phase
	switch phase {
	case mariav1alpha1.JobPhaseSucceeded, mariav1alpha1.JobPhaseFailed, mariav1alpha1.JobPhaseCancelled:
		// Completed, check TTL
		if instance.Spec.TTLSecondsAfterFinished != nil {
			return ctrl.Result{RequeueAfter: time.Duration(*instance.Spec.TTLSecondsAfterFinished) * time.Second}, nil
		}
		return ctrl.Result{}, nil
	case mariav1alpha1.JobPhasePending:
		return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
	default:
		return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
	}
}

// buildJobPayload builds the HTTP payload for job submission
func (r *MarieJobReconciler) buildJobPayload(instance *mariav1alpha1.MarieJob) map[string]interface{} {
	payload := map[string]interface{}{
		"job_type":   instance.Spec.JobType,
		"priority":   instance.Spec.Priority,
		"entrypoint": instance.Spec.Entrypoint,
	}

	if instance.Spec.Input != nil {
		input := map[string]interface{}{}
		if instance.Spec.Input.URI != "" {
			input["uri"] = instance.Spec.Input.URI
		}
		if instance.Spec.Input.Data != "" {
			input["data"] = instance.Spec.Input.Data
		}
		if instance.Spec.Input.ContentType != "" {
			input["content_type"] = instance.Spec.Input.ContentType
		}
		payload["input"] = input
	}

	if instance.Spec.Output != nil {
		output := map[string]interface{}{}
		if instance.Spec.Output.URI != "" {
			output["uri"] = instance.Spec.Output.URI
		}
		if instance.Spec.Output.Format != "" {
			output["format"] = instance.Spec.Output.Format
		}
		payload["output"] = output
	}

	if instance.Spec.Options != nil {
		payload["options"] = instance.Spec.Options
	}

	if instance.Spec.SLA != nil {
		sla := map[string]interface{}{}
		if instance.Spec.SLA.SoftDeadline != "" {
			sla["soft_deadline"] = instance.Spec.SLA.SoftDeadline
		}
		if instance.Spec.SLA.HardDeadline != "" {
			sla["hard_deadline"] = instance.Spec.SLA.HardDeadline
		}
		payload["sla"] = sla
	}

	return payload
}

// getInputURI extracts the input URI from the job spec
func getInputURI(instance *mariav1alpha1.MarieJob) string {
	if instance.Spec.Input != nil && instance.Spec.Input.URI != "" {
		return instance.Spec.Input.URI
	}
	return ""
}

// SetupWithManager sets up the controller with the Manager
func (r *MarieJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&mariav1alpha1.MarieJob{}).
		Owns(&batchv1.Job{}).
		Complete(r)
}
