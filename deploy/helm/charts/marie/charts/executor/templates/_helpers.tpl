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

{{- define "executor.headlessNamingVersion" -}}
{{- $version := "v1" -}}
{{- if hasKey .Values "headlessServiceNamingVersion" -}}
{{- $version = .Values.headlessServiceNamingVersion -}}
{{- end -}}
{{- if or (not (kindIs "string" $version)) (not (has $version (list "v1" "v2"))) -}}
{{- fail "executor identity: unsupported headless Service naming version" -}}
{{- end -}}
{{- $version -}}
{{- end -}}

{{- define "executor.headlessServiceName" -}}
{{- $version := include "executor.headlessNamingVersion" .root -}}
{{- if eq $version "v1" -}}
{{- include "executor.poolFullname" (list .pool .root.Release.Name) -}}-headless
{{- else -}}
{{- $identity := toJson (list "marie.executor-headless/v2" .root.Release.Namespace .root.Release.Name .pool.name) -}}
{{- printf "ex-v2-%s-headless" (sha256sum $identity | trunc 24) -}}
{{- end -}}
{{- end -}}

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
marie.ai/pool: {{ $pool.name | quote }}
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
{{- $image := default dict $pool.image }}
{{- $registry := $global.imageRegistry | default "" -}}
{{- $repository := $image.repository | default $global.marie.image.repository | default "marieai/marie" -}}
{{- $tag := $image.tag | default $global.marie.image.tag | default $appVersion -}}
{{- $digest := $image.digest | default $global.marie.image.digest | default "" -}}
{{- $reference := $repository -}}
{{- if $registry }}
{{- $reference = printf "%s/%s" $registry $repository -}}
{{- end }}
{{- if $digest }}
{{- printf "%s@%s" $reference $digest -}}
{{- else }}
{{- printf "%s:%s" $reference $tag -}}
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
