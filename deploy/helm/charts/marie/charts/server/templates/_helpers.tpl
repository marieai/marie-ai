{{/*
Server name
*/}}
{{- define "server.name" -}}
{{- default "server" .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Server fullname
*/}}
{{- define "server.fullname" -}}
{{- $name := default "server" .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Server labels
*/}}
{{- define "server.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "server.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: server
{{- end }}

{{/*
Server selector labels
*/}}
{{- define "server.selectorLabels" -}}
app.kubernetes.io/name: {{ include "server.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Server image
*/}}
{{- define "server.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" -}}
{{- $repository := .Values.image.repository | default "marieai/marie" -}}
{{- $tag := .Values.image.tag | default .Values.global.marie.image.tag | default .Chart.AppVersion -}}
{{- $digest := .Values.image.digest | default .Values.global.marie.image.digest | default "" -}}
{{- $image := $repository -}}
{{- if $registry }}
{{- $image = printf "%s/%s" $registry $repository -}}
{{- end }}
{{- if $digest }}
{{- printf "%s@%s" $image $digest -}}
{{- else }}
{{- printf "%s:%s" $image $tag -}}
{{- end }}
{{- end }}

{{/*
Server service account name
*/}}
{{- define "server.serviceAccountName" -}}
{{- if .Values.global.serviceAccount.create }}
{{- default (printf "%s" .Release.Name) .Values.global.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.global.serviceAccount.name }}
{{- end }}
{{- end }}
