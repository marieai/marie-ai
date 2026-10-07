{{- define "otel-collector.name" -}}
{{- default "otel-collector" .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "otel-collector.fullname" -}}
{{- $name := default "otel-collector" .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "otel-collector.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "otel-collector.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: otel-collector
{{- end }}

{{- define "otel-collector.selectorLabels" -}}
app.kubernetes.io/name: {{ include "otel-collector.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "otel-collector.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" -}}
{{- $repository := .Values.image.repository | default "otel/opentelemetry-collector-contrib" -}}
{{- $tag := .Values.image.tag | default "latest" -}}
{{- if .Values.image.digest -}}
{{- if $registry -}}
{{- printf "%s/%s@%s" $registry $repository .Values.image.digest -}}
{{- else -}}
{{- printf "%s@%s" $repository .Values.image.digest -}}
{{- end -}}
{{- else -}}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- else }}
{{- printf "%s:%s" $repository $tag -}}
{{- end }}
{{- end }}
{{- end }}

{{/*
ClickHouse host: the configured one, or this release's bundled ClickHouse.
*/}}
{{- define "otel-collector.clickhouseHost" -}}
{{- .Values.clickhouse.host | default (printf "%s-clickhouse" .Release.Name) -}}
{{- end }}
