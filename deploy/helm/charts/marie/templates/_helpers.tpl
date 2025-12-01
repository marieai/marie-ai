{{/*
Expand the name of the chart.
*/}}
{{- define "marie.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "marie.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "marie.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "marie.labels" -}}
helm.sh/chart: {{ include "marie.chart" . }}
{{ include "marie.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.global.labels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "marie.selectorLabels" -}}
app.kubernetes.io/name: {{ include "marie.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "marie.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "marie.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper Marie image name
*/}}
{{- define "marie.image" -}}
{{- $registryName := .Values.global.imageRegistry -}}
{{- $repositoryName := .Values.marie.image.repository -}}
{{- $tag := .Values.marie.image.tag | default .Chart.AppVersion -}}
{{- if $registryName }}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else }}
{{- printf "%s:%s" $repositoryName $tag -}}
{{- end }}
{{- end }}

{{/*
Return the proper image pull secrets
*/}}
{{- define "marie.imagePullSecrets" -}}
{{- if .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range .Values.global.imagePullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
PostgreSQL host
*/}}
{{- define "marie.postgresql.host" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" (include "marie.fullname" .) }}
{{- else }}
{{- .Values.postgresql.external.host }}
{{- end }}
{{- end }}

{{/*
PostgreSQL port
*/}}
{{- define "marie.postgresql.port" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "5432" }}
{{- else }}
{{- .Values.postgresql.external.port | default 5432 | toString }}
{{- end }}
{{- end }}

{{/*
PostgreSQL database
*/}}
{{- define "marie.postgresql.database" -}}
{{- if .Values.postgresql.enabled }}
{{- .Values.postgresql.auth.database }}
{{- else }}
{{- .Values.postgresql.external.database }}
{{- end }}
{{- end }}

{{/*
PostgreSQL secret name
*/}}
{{- define "marie.postgresql.secretName" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" (include "marie.fullname" .) }}
{{- else }}
{{- .Values.postgresql.external.existingSecret }}
{{- end }}
{{- end }}

{{/*
RabbitMQ host
*/}}
{{- define "marie.rabbitmq.host" -}}
{{- if .Values.rabbitmq.enabled }}
{{- printf "%s-rabbitmq" (include "marie.fullname" .) }}
{{- else }}
{{- .Values.rabbitmq.external.host }}
{{- end }}
{{- end }}

{{/*
RabbitMQ port
*/}}
{{- define "marie.rabbitmq.port" -}}
{{- if .Values.rabbitmq.enabled }}
{{- printf "5672" }}
{{- else }}
{{- .Values.rabbitmq.external.port | default 5672 | toString }}
{{- end }}
{{- end }}

{{/*
etcd endpoints
*/}}
{{- define "marie.etcd.endpoints" -}}
{{- if .Values.etcd.enabled }}
{{- $replicaCount := int .Values.etcd.replicaCount }}
{{- $fullname := include "marie.fullname" . }}
{{- $endpoints := list }}
{{- range $i := until $replicaCount }}
{{- $endpoints = append $endpoints (printf "http://%s-etcd-%d.%s-etcd-headless:2379" $fullname $i $fullname) }}
{{- end }}
{{- join "," $endpoints }}
{{- else }}
{{- join "," .Values.etcd.external.endpoints }}
{{- end }}
{{- end }}

{{/*
Common environment variables for database connection
*/}}
{{- define "marie.databaseEnv" -}}
- name: DATABASE_HOST
  value: {{ include "marie.postgresql.host" . | quote }}
- name: DATABASE_PORT
  value: {{ include "marie.postgresql.port" . | quote }}
- name: DATABASE_NAME
  value: {{ include "marie.postgresql.database" . | quote }}
- name: DATABASE_USER
  {{- if .Values.postgresql.enabled }}
  value: {{ .Values.postgresql.auth.username | quote }}
  {{- else }}
  value: {{ .Values.postgresql.external.username | quote }}
  {{- end }}
- name: DATABASE_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "marie.postgresql.secretName" . }}
      key: {{ if .Values.postgresql.enabled }}password{{ else }}{{ .Values.postgresql.external.existingSecretPasswordKey }}{{ end }}
{{- end }}

{{/*
Common environment variables for RabbitMQ connection
*/}}
{{- define "marie.rabbitmqEnv" -}}
- name: RABBITMQ_HOST
  value: {{ include "marie.rabbitmq.host" . | quote }}
- name: RABBITMQ_PORT
  value: {{ include "marie.rabbitmq.port" . | quote }}
- name: RABBITMQ_USER
  {{- if .Values.rabbitmq.enabled }}
  value: {{ .Values.rabbitmq.auth.username | quote }}
  {{- else }}
  value: {{ .Values.rabbitmq.external.username | quote }}
  {{- end }}
- name: RABBITMQ_PASSWORD
  valueFrom:
    secretKeyRef:
      {{- if .Values.rabbitmq.enabled }}
      name: {{ include "marie.fullname" . }}-rabbitmq
      key: rabbitmq-password
      {{- else }}
      name: {{ .Values.rabbitmq.external.existingSecret }}
      key: {{ .Values.rabbitmq.external.existingSecretPasswordKey }}
      {{- end }}
{{- end }}

{{/*
Common environment variables for etcd connection
*/}}
{{- define "marie.etcdEnv" -}}
- name: ETCD_ENDPOINTS
  value: {{ include "marie.etcd.endpoints" . | quote }}
{{- end }}

{{/*
Storage environment variables
*/}}
{{- define "marie.storageEnv" -}}
- name: STORAGE_TYPE
  value: {{ .Values.storage.type | quote }}
{{- if eq .Values.storage.type "s3" }}
- name: S3_BUCKET
  value: {{ .Values.storage.s3.bucket | quote }}
- name: S3_REGION
  value: {{ .Values.storage.s3.region | quote }}
{{- if .Values.storage.s3.endpoint }}
- name: S3_ENDPOINT
  value: {{ .Values.storage.s3.endpoint | quote }}
{{- end }}
{{- if .Values.storage.s3.accessKey }}
- name: AWS_ACCESS_KEY_ID
  value: {{ .Values.storage.s3.accessKey | quote }}
- name: AWS_SECRET_ACCESS_KEY
  value: {{ .Values.storage.s3.secretKey | quote }}
{{- else if .Values.storage.s3.existingSecret }}
- name: AWS_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: {{ .Values.storage.s3.existingSecret }}
      key: access-key
- name: AWS_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.storage.s3.existingSecret }}
      key: secret-key
{{- end }}
{{- else if eq .Values.storage.type "gcs" }}
- name: GCS_BUCKET
  value: {{ .Values.storage.gcs.bucket | quote }}
- name: GCS_PROJECT
  value: {{ .Values.storage.gcs.project | quote }}
{{- end }}
{{- end }}

{{/*
Tracing environment variables
*/}}
{{- define "marie.tracingEnv" -}}
{{- if .Values.marie.tracing.enabled }}
- name: TRACING_ENABLED
  value: "true"
- name: TRACING_EXPORTER
  value: {{ .Values.marie.tracing.exporterType | quote }}
{{- if .Values.observability.jaeger.enabled }}
- name: JAEGER_AGENT_HOST
  value: {{ .Values.observability.jaeger.agent.host | quote }}
- name: JAEGER_AGENT_PORT
  value: {{ .Values.observability.jaeger.agent.port | default 6831 | quote }}
{{- end }}
{{- if .Values.observability.otel.enabled }}
- name: OTEL_EXPORTER_OTLP_ENDPOINT
  value: {{ .Values.observability.otel.endpoint | quote }}
- name: OTEL_EXPORTER_OTLP_INSECURE
  value: {{ .Values.observability.otel.insecure | quote }}
{{- end }}
{{- end }}
{{- end }}
