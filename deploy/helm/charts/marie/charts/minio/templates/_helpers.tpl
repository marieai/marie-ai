{{/*
MinIO name
*/}}
{{- define "minio.name" -}}
{{- default "minio" .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
MinIO fullname
*/}}
{{- define "minio.fullname" -}}
{{- $name := default "minio" .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
MinIO labels
*/}}
{{- define "minio.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "minio.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
MinIO selector labels
*/}}
{{- define "minio.selectorLabels" -}}
app.kubernetes.io/name: {{ include "minio.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
MinIO image
*/}}
{{- define "minio.image" -}}
{{- $registry := .Values.image.registry | default "docker.io" -}}
{{- printf "%s/%s:%s" $registry .Values.image.repository .Values.image.tag -}}
{{- end }}

{{- define "minio.mcImage" -}}
{{- $registry := .Values.provisioning.image.registry | default "docker.io" -}}
{{- printf "%s/%s:%s" $registry .Values.provisioning.image.repository .Values.provisioning.image.tag -}}
{{- end }}

{{/*
MinIO credentials secret name
*/}}
{{- define "minio.secretName" -}}
{{- default (include "minio.fullname" .) .Values.auth.existingSecret }}
{{- end }}
