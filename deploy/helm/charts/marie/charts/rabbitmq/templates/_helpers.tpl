{{/*
RabbitMQ name
*/}}
{{- define "rabbitmq.name" -}}
{{- default "rabbitmq" .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
RabbitMQ fullname
*/}}
{{- define "rabbitmq.fullname" -}}
{{- $name := default "rabbitmq" .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
RabbitMQ labels
*/}}
{{- define "rabbitmq.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "rabbitmq.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
RabbitMQ selector labels
*/}}
{{- define "rabbitmq.selectorLabels" -}}
app.kubernetes.io/name: {{ include "rabbitmq.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
RabbitMQ image
*/}}
{{- define "rabbitmq.image" -}}
{{- $registry := .Values.image.registry | default "docker.io" -}}
{{- if .Values.image.digest -}}
{{- printf "%s/%s@%s" $registry .Values.image.repository .Values.image.digest -}}
{{- else -}}
{{- printf "%s/%s:%s" $registry .Values.image.repository .Values.image.tag -}}
{{- end -}}
{{- end }}

{{/*
RabbitMQ password secret name
*/}}
{{- define "rabbitmq.secretName" -}}
{{- default (include "rabbitmq.fullname" .) .Values.auth.existingSecret }}
{{- end }}
