Batch processing pipeline for OCR document extraction using Marie AI services.

---
`batch_document_ocr` automates end-to-end extraction of text and metadata from scanned documents. It supports both **single-file** and **directory-based batch processing**, seamlessly integrating:

- **S3 storage uploads**
- **OCR job submission to Marie queues**
- **Event-driven retrieval of results**
- **Structured JSON output storage**

---

## Supported File Types

- TIFF (`.tif`, `.tiff`)
- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)

---

Downloaded assets for generators are downloaded to `~/.marie/generators`

```markdown
~/.marie/generators
├── 5b251fb33519e6828c25df663d7624b1
│  ├── adlib
│  ├── assets
│  ├── blobs
│  ├── burst
│  ├── clean
│  ├── pdf
│  └── results
```
and an individual meta file is stored in the `otuput_dir`, this is just for convininence.


## Command-Line Usage

```bash
python batch_document_ocr.py \
  --input <input_directory_or_file> \
  --output-dir <output_directory> \
  --config <config_json>
```

### Arguments

- `--input`  
  Path to a file or directory containing images.

- `--output-dir`  
  Directory where output JSON files will be stored.

- `--config`  
  JSON configuration file with storage and queue settings.

---

## Processing Flow

1. **Upload**  
   Each document is uploaded to S3.

2. **Job Submission**  
   A job request is posted to the Marie API queue (`extract` or `gen5_extract`).

3. **Event Listening**  
   The script listens for `"extract.completed"` messages.

4. **Result Retrieval**  
   OCR results are downloaded and saved as JSON.

---

## Example

Batch process a directory:

```bash
python ./batch_document_ocr.py --config config.dev.json  \
  --pipeline default --input ~/datasets/private/SAMPLE-001/images \
  --output_dir ~/datasets/private/SAMPLE-001/annotations
```
---

## Configuration

The `config.dev.json` file must include:
```json
{
  "api_base_url": "http://127.0.0.1:51000",
  "api_key": "mau_t6qDi1BcL1NkLI8I6iM8z1va0nZP01UQ6LWecpbDz6mbxWgIIIZPfQ",
  "default_queue_id": "0000-0000-0000-0000",
  "storage": {
    "S3_ACCESS_KEY_ID": "MARIEACCESSKEY",
    "S3_SECRET_ACCESS_KEY": "MARIESECRETACCESSKEY",
    "S3_STORAGE_BUCKET_NAME": "marie",
    "S3_ENDPOINT_URL": "http://localhost:8000",
    "S3_ADDRESSING_STYLE": "path"
  },
  "queue": {
    "hostname": "localhost",
    "port": 5672,
    "username": "guest",
    "password": "guest"
  }
}
```

---

```shell
python ./batch_document_ocr.py 

usage: batch_document_ocr.py [-h] --config CONFIG --input INPUT --output_dir OUTPUT_DIR --pipeline PIPELINE
batch_document_ocr.py: error: the following arguments are required: --config, --input, --output_dir, --pipeline
```

```markdown
python ./batch_document_ocr.py --config config.dev.json  --pipeline default --input ~/datasets/private/SAMPLE-001/images --output_dir ~/datasets/private/SAMPLE-001/annotations
```

Sample output:

```markdown
INFO   2025-07-09 06:33:40,584:            : MARIE@781288 Ref ID: 148443671_0.png, Ref Type: extract                                                                                                                                          
INFO   2025-07-09 06:33:40,588:            : MARIE@781288 Downloading results: s3://marie/extract/148443671_0/148443671_0.png.meta.json to /home/greg/datasets/private/SAMPLE-001/annotations/148443671_0.json                                
INFO   2025-07-09 06:33:40,662:            : MARIE@781288 Extracted hash: c79cb3540c6d55552f94bcc46d2adf12                                                                                                                                    
INFO   2025-07-09 06:33:40,663:            : MARIE@781288 Copying s3://marie/extract/148443671_0 to /home/greg/.marie/generators/c79cb3540c6d55552f94bcc46d2adf12                                                                             
INFO   2025-07-09 06:33:40,709:            : MARIE@781288 Copying completed                                                                                                                                                                   
INFO   2025-07-09 06:33:41,038:            : MARIE@781288 Message handler: queue size=1, event=extract.completed, jobid=0686e539-311c-7a4b-8000-5ec141b76508                                                               [07/09/25 06:33:41]
INFO   2025-07-09 06:33:41,038:            : MARIE@781288 Processing completed message: {'api_key': 'mau_t6qDi1BcL1NkLI8I6iM8z1va0nZP01UQ6LWecpbDz6mbxWgIIIZPfQ', 'jobid': '0686e539-311c-7a4b-8000-5ec141b76508',                            
       'event': 'extract.completed', 'jobtag': 'extract', 'status': 'OK', 'timestamp': 1752060821, 'payload': '{\n  "on": "extract_executor://document/extract",\n  "uri":                                                                    
       "s3://marie/extract/148444817_2/148444817_2.png",\n  "name": "default",\n  "type": "pipeline",\n  "doc_id": "148444817_2.png",\n  "policy": "allow_all",\n  "ref_id": "148444817_2.png",\n  "planner": "extract",\n                    
       "doc_type": "extract",\n  "hard_sla": "2025-07-09T12:33:39.047452",\n  "ref_type": "extract",\n  "soft_sla": "2025-07-09T08:33:39.047452",\n  "project_id":                                                                            
       "mau_t6qDi1BcL1NkLI8I6iM8z1va0nZP01UQ6LWecpbDz6mbxWgIIIZPfQ",\n  "page_cleaner": {\n    "enabled": false\n  },\n  "page_boundary": {\n    "enabled": false\n  },\n  "page_splitter": {\n    "enabled": false\n                         
       },\n  "page_classifier": {\n    "enabled": false\n  },\n  "template_matching": {\n    "enabled": false,\n    "definition_id": "0"\n  }\n}'}                                                                                            
INFO   2025-07-09 06:33:41,039:            : MARIE@781288 Ref ID: 148444817_2.png, Ref Type: extract                                                                                                                                          
INFO   2025-07-09 06:33:41,043:            : MARIE@781288 Downloading results: s3://marie/extract/148444817_2/148444817_2.png.meta.json to /home/greg/datasets/private/SAMPLE-001/annotations/148444817_2.json                                
INFO   2025-07-09 06:33:41,113:            : MARIE@781288 Extracted hash: bde9eece5ba1e28456e033ffab17500d                                                                                                                                    
INFO   2025-07-09 06:33:41,114:            : MARIE@781288 Copying s3://marie/extract/148444817_2 to /home/greg/.marie/generators/bde9eece5ba1e28456e033ffab17500d                                                                             
INFO   2025-07-09 06:33:41,164:            : MARIE@781288 Copying completed                                                                                                                                                                   
```