{{- define "hyperdx.name" -}}
{{- default "hyperdx" .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "hyperdx.fullname" -}}
{{- $name := default "hyperdx" .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "hyperdx.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "hyperdx.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: hyperdx
{{- end }}

{{- define "hyperdx.selectorLabels" -}}
app.kubernetes.io/name: {{ include "hyperdx.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "hyperdx.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" -}}
{{- $repository := .Values.image.repository | default "docker.hyperdx.io/hyperdx/hyperdx" -}}
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
ClickHouse connection HyperDX creates for the first team (DEFAULT_CONNECTIONS).
*/}}
{{- define "hyperdx.defaultConnections" -}}
{{- $host := .Values.clickhouse.host | default (printf "%s-clickhouse" .Release.Name) -}}
{{- list (dict "name" "Default" "host" (printf "http://%s:%v" $host .Values.clickhouse.httpPort) "username" .Values.clickhouse.username "password" .Values.clickhouse.password) | toJson -}}
{{- end }}

{{/*
Log, trace and metric sources over the "otel" tables (DEFAULT_SOURCES). Sources refer
to the connection and to each other by name; HyperDX resolves the names to ids.
*/}}
{{- define "hyperdx.defaultSources" -}}
{{- $db := .Values.clickhouse.database -}}
{{- $logs := dict
  "name" "Logs" "kind" "log" "connection" "Default"
  "from" (dict "databaseName" $db "tableName" "otel_logs")
  "timestampValueExpression" "Timestamp"
  "displayedTimestampValueExpression" "Timestamp"
  "implicitColumnExpression" "Body"
  "bodyExpression" "Body"
  "severityTextExpression" "SeverityText"
  "serviceNameExpression" "ServiceName"
  "eventAttributesExpression" "LogAttributes"
  "resourceAttributesExpression" "ResourceAttributes"
  "traceIdExpression" "TraceId"
  "spanIdExpression" "SpanId"
  "defaultTableSelectExpression" "Timestamp, ServiceName as service, SeverityText as level, Body"
  "traceSourceId" "Traces" "metricSourceId" "Metrics" -}}
{{- $traces := dict
  "name" "Traces" "kind" "trace" "connection" "Default"
  "from" (dict "databaseName" $db "tableName" "otel_traces")
  "timestampValueExpression" "Timestamp"
  "displayedTimestampValueExpression" "Timestamp"
  "implicitColumnExpression" "SpanName"
  "spanNameExpression" "SpanName"
  "spanKindExpression" "SpanKind"
  "serviceNameExpression" "ServiceName"
  "durationExpression" "Duration"
  "durationPrecision" 9
  "statusCodeExpression" "StatusCode"
  "statusMessageExpression" "StatusMessage"
  "traceIdExpression" "TraceId"
  "spanIdExpression" "SpanId"
  "parentSpanIdExpression" "ParentSpanId"
  "eventAttributesExpression" "SpanAttributes"
  "resourceAttributesExpression" "ResourceAttributes"
  "spanEventsValueExpression" "Events"
  "spanLinksValueExpression" "Links"
  "defaultTableSelectExpression" "Timestamp, ServiceName as service, StatusCode as level, round(Duration / 1e6) as duration, SpanName"
  "logSourceId" "Logs" "metricSourceId" "Metrics" -}}
{{- $metrics := dict
  "name" "Metrics" "kind" "metric" "connection" "Default"
  "from" (dict "databaseName" $db "tableName" "")
  "timestampValueExpression" "TimeUnix"
  "resourceAttributesExpression" "ResourceAttributes"
  "metricTables" (dict "gauge" "otel_metrics_gauge" "histogram" "otel_metrics_histogram" "sum" "otel_metrics_sum" "summary" "otel_metrics_summary" "exponential histogram" "otel_metrics_exp_histogram") -}}
{{- list $logs $traces $metrics | toJson -}}
{{- end }}
