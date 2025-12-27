{{/*
ClickHouse name
*/}}
{{- define "clickhouse.name" -}}
{{- default "clickhouse" .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
ClickHouse fullname
*/}}
{{- define "clickhouse.fullname" -}}
{{- $name := default "clickhouse" .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
ClickHouse labels
*/}}
{{- define "clickhouse.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "clickhouse.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: clickhouse
{{- end }}

{{/*
ClickHouse selector labels
*/}}
{{- define "clickhouse.selectorLabels" -}}
app.kubernetes.io/name: {{ include "clickhouse.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
ClickHouse image
*/}}
{{- define "clickhouse.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" -}}
{{- $repository := .Values.image.repository | default "clickhouse/clickhouse-server" -}}
{{- $tag := .Values.image.tag | default "latest" -}}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- else }}
{{- printf "%s:%s" $repository $tag -}}
{{- end }}
{{- end }}

{{/*
ClickHouse service account name
*/}}
{{- define "clickhouse.serviceAccountName" -}}
{{- if .Values.global.serviceAccount.create }}
{{- default (printf "%s" .Release.Name) .Values.global.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.global.serviceAccount.name }}
{{- end }}
{{- end }}
