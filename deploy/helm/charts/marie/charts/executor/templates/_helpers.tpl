{{/*
Executor base name
*/}}
{{- define "executor.name" -}}
{{- default "executor" .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Executor fullname for a pool
*/}}
{{- define "executor.poolFullname" -}}
{{- $pool := index . 0 }}
{{- $release := index . 1 }}
{{- printf "%s-executor-%s" $release $pool.name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Executor labels
*/}}
{{- define "executor.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: executor
{{- end }}

{{/*
Executor selector labels for a pool
*/}}
{{- define "executor.poolSelectorLabels" -}}
{{- $pool := index . 0 }}
{{- $release := index . 1 }}
app.kubernetes.io/name: executor
app.kubernetes.io/instance: {{ $release }}
marie.ai/pool: {{ $pool.name }}
{{- end }}

{{/*
Executor pool labels
*/}}
{{- define "executor.poolLabels" -}}
{{- $pool := index . 0 }}
{{- $ctx := index . 1 }}
helm.sh/chart: {{ printf "%s-%s" $ctx.Chart.Name $ctx.Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "executor.poolSelectorLabels" (list $pool $ctx.Release.Name) }}
app.kubernetes.io/version: {{ $ctx.Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ $ctx.Release.Service }}
app.kubernetes.io/component: executor
{{- if $pool.gpu }}
marie.ai/gpu: "true"
{{- end }}
{{- end }}

{{/*
Executor image for a pool
*/}}
{{- define "executor.poolImage" -}}
{{- $pool := index . 0 }}
{{- $global := index . 1 }}
{{- $appVersion := index . 2 }}
{{- $registry := $global.imageRegistry | default "" -}}
{{- $repository := $pool.image.repository | default $global.marie.image.repository | default "marieai/marie" -}}
{{- $tag := $pool.image.tag | default $global.marie.image.tag | default $appVersion -}}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- else }}
{{- printf "%s:%s" $repository $tag -}}
{{- end }}
{{- end }}

{{/*
Executor service account name
*/}}
{{- define "executor.serviceAccountName" -}}
{{- if .Values.global.serviceAccount.create }}
{{- default (printf "%s" .Release.Name) .Values.global.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.global.serviceAccount.name }}
{{- end }}
{{- end }}
