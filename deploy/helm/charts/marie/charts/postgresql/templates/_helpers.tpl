{{- define "postgresql.name" -}}
{{- default "postgresql" .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "postgresql.fullname" -}}
{{- $name := default "postgresql" .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "postgresql.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "postgresql.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "postgresql.selectorLabels" -}}
app.kubernetes.io/name: {{ include "postgresql.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "postgresql.image" -}}
{{- $registry := .Values.image.registry | default "ghcr.io" -}}
{{- if .Values.image.digest -}}
{{- printf "%s/%s@%s" $registry .Values.image.repository .Values.image.digest -}}
{{- else -}}
{{- printf "%s/%s:%s" $registry .Values.image.repository .Values.image.tag -}}
{{- end -}}
{{- end }}

{{- define "postgresql.secretName" -}}
{{- default (include "postgresql.fullname" .) .Values.auth.existingSecret }}
{{- end }}

{{- define "postgresql.password" -}}
{{- .Values.auth.password | default .Values.auth.postgresPassword | default "marie-password" -}}
{{- end }}

{{- define "postgresql.postgresPassword" -}}
{{- .Values.auth.postgresPassword | default .Values.auth.password | default "marie-password" -}}
{{- end }}
