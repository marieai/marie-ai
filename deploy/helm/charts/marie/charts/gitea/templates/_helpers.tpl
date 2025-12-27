{{/*
Gitea name
*/}}
{{- define "gitea.name" -}}
{{- default "gitea" .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Gitea fullname
*/}}
{{- define "gitea.fullname" -}}
{{- $name := default "gitea" .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Gitea labels
*/}}
{{- define "gitea.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "gitea.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: gitea
{{- end }}

{{/*
Gitea selector labels
*/}}
{{- define "gitea.selectorLabels" -}}
app.kubernetes.io/name: {{ include "gitea.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Gitea image
*/}}
{{- define "gitea.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" -}}
{{- $repository := .Values.image.repository | default "gitea/gitea" -}}
{{- $tag := .Values.image.tag | default "latest" -}}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- else }}
{{- printf "%s:%s" $repository $tag -}}
{{- end }}
{{- end }}

{{/*
Gitea service account name
*/}}
{{- define "gitea.serviceAccountName" -}}
{{- if .Values.global.serviceAccount.create }}
{{- default (printf "%s" .Release.Name) .Values.global.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.global.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Gitea database host
*/}}
{{- define "gitea.databaseHost" -}}
{{- if .Values.database.host }}
{{- .Values.database.host }}
{{- else if .Values.global.postgresql }}
{{- printf "%s-postgresql" .Release.Name }}
{{- else }}
{{- "localhost" }}
{{- end }}
{{- end }}

{{/*
Gitea database user
*/}}
{{- define "gitea.databaseUser" -}}
{{- if .Values.database.user }}
{{- .Values.database.user }}
{{- else if .Values.global.postgresql }}
{{- .Values.global.postgresql.auth.username | default "postgres" }}
{{- else }}
{{- "gitea" }}
{{- end }}
{{- end }}
