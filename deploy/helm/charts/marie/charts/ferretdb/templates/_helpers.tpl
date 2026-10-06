{{- define "ferretdb.name" -}}
{{- default "ferretdb" .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "ferretdb.fullname" -}}
{{- $name := default "ferretdb" .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "ferretdb.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "ferretdb.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: ferretdb
{{- end }}

{{- define "ferretdb.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ferretdb.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "ferretdb.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" -}}
{{- $repository := .Values.image.repository | default "ghcr.io/ferretdb/ferretdb" -}}
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
