{{- define "etcd.name" -}}
{{- default "etcd" .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "etcd.fullname" -}}
{{- $name := default "etcd" .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "etcd.headlessName" -}}
{{- printf "%s-headless" (include "etcd.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "etcd.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "etcd.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "etcd.selectorLabels" -}}
app.kubernetes.io/name: {{ include "etcd.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "etcd.image" -}}
{{- $registry := .Values.image.registry | default "quay.io" -}}
{{- printf "%s/%s:%s" $registry .Values.image.repository .Values.image.tag -}}
{{- end }}
