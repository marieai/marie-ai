---
sidebar_position: 1
---

# Document Pipelines

Document pipelines are the core of the document processing system. They are responsible for processing documents and
extracting information from them.

## Basic Concepts

## Pipeline Components

## Pipeline Configuration

```yaml
executors:
  - name: extract_t
    uses:
      jtype: TextExtractionExecutor
      with:
        storage:
          # postgresql configuration. Will be used only if value of backend is "psql"
          psql:
            <<: *psql_conf_shared
            default_table: extract_metadata
            enabled: True
        pipeline:
          name: 'default'
          page_classifier:
            - model_name_or_path: 'marie/lmv3-medical-document-classification'
              type: 'transformers'
              device: 'cuda'
              enabled: True
              name: 'medical_page_classifier'
            - model_name_or_path: 'marie/lmv3-medical-document-payer'
              type: 'transformers'
              enabled: True
              device: 'cuda'
              name: 'medical_payer_classifier'
          page_indexer:
            - model_name_or_path: 'marie/layoutlmv3-medical-document-indexer'
              enabled: True
              type: 'transformers'
              device: 'cuda'
              name: 'page_indexer_patient'
              filter:
                type: 'regex'
                pattern: '.*'
          page_splitter:
            model_name_or_path: 'marie/layoutlmv3-medical-document-splitter'
            enabled: True
      metas:
        py_modules:
          - marie.executor.text
    timeout_ready: 3000000
    replicas: 1
    #    replicas: ${{ CONTEXT.gpu_device_count }}
    env:
      CUDA_VISIBLE_DEVICES: RR

```

## Runtime Configuration

Each pipeline can be configured at runtime. This allows for the configuration of the pipeline to be changed without
restarting the application and without affecting other pipelines.

Sample payload configuration to be sent to the pipeline with the `POST /api/extract` endpoint:

```json
{
  "queue_id": "0000-0000-0000-0000",
  "data": "s3|base64||url",
  "mode": "MULTILINE|RAW|SPARSE|LINE",
  "doc_id": "1234",
  "doc_type": "extract-adhoc",
  "features": [
    {
      "type": "pipeline",
      "name": "default",
      "page_classifier": {
        "enabled": true
      },
      "page_splitter": {
        "enabled": true
      },
      "overlay": {
        "enabled": false
      },
      "ocr": {
        "document": {
          "engine": "default"
        },
        "region": {
          "engine": "best"
        }
      }
    }
  ]
}

```
