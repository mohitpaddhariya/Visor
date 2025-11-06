# OCR Job Queue API Documentation

This document describes the Modal job queue-based OCR API with Server-Sent Events (SSE) support for real-time progress updates.

## Overview

The job queue system allows you to:
- Submit long-running OCR tasks asynchronously
- Get immediate job IDs instead of waiting for completion
- Monitor job progress in real-time via Server-Sent Events (SSE)
- Poll job status and retrieve results when ready
- Handle multiple concurrent jobs efficiently

## Architecture

```
Client                    FastAPI Server              Modal Functions
  |                             |                            |
  |-- POST /ocr/submit -------->|                            |
  |                             |-- spawn() ---------------->|
  |<-- {job_id} ---------------|                            |
  |                             |                            |
  |-- GET /ocr/stream/{id} --->|                            |
  |<== SSE: status events ===<=|-- poll get() ------------->|
  |<== SSE: result ========<===|<-- result ----------------|
  |                             |                            |
  |-- GET /ocr/result/{id} --->|                            |
  |<-- {full_results} ---------|                            |
```

## API Endpoints

### 1. Submit OCR Job

**POST** `/ocr/submit`

Submit a new OCR job to the queue.

#### Parameters:
- `file` (form-data): PDF file to process
- `dpi` (query, optional): DPI for rendering (72-600, default: 200)
- `model` (query, optional): OCR model to use (`dotsocr` or `lightonocr`)

#### Response:
```json
{
  "success": true,
  "job_id": "fc-01JCFZ8X...",
  "filename": "document.pdf",
  "total_pages": 5,
  "model": "dotsocr",
  "dpi": 200,
  "message": "Job submitted successfully. Use /ocr/status/{job_id} to check status or /ocr/stream/{job_id} for real-time updates.",
  "submitted_at": "2025-11-06T10:30:00.000Z"
}
```

#### Example (curl):
```bash
curl -X POST "http://localhost:8003/ocr/submit?model=dotsocr&dpi=200" \
  -F "file=@document.pdf"
```

#### Example (JavaScript):
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('/ocr/submit?model=dotsocr&dpi=200', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log('Job ID:', data.job_id);
```

---

### 2. Stream Job Progress (SSE)

**GET** `/ocr/stream/{job_id}`

Stream real-time job progress updates via Server-Sent Events.

#### SSE Events:

**Event: `status`**
```json
{
  "status": "pending|running|completed",
  "job_id": "fc-01JCFZ8X...",
  "elapsed": 15
}
```

**Event: `result`**
```json
{
  "job_id": "fc-01JCFZ8X...",
  "status": "completed",
  "total_pages": 5,
  "model": "dotsocr",
  "results": {
    "page_0": {"status": "success", "data": [...]},
    "page_1": {"status": "success", "data": [...]}
  },
  "metadata": {
    "filename": "document.pdf",
    "dpi": 200
  }
}
```

**Event: `error`**
```json
{
  "status": "error",
  "job_id": "fc-01JCFZ8X...",
  "error": "Error message"
}
```

**Event: `timeout`**
```json
{
  "status": "timeout",
  "job_id": "fc-01JCFZ8X...",
  "message": "Job still running after 10 minutes"
}
```

**Event: `close`**
```json
{
  "message": "Stream complete"
}
```

#### Example (JavaScript EventSource):
```javascript
const eventSource = new EventSource(`/ocr/stream/${jobId}`);

eventSource.addEventListener('status', (e) => {
  const data = JSON.parse(e.data);
  console.log('Status:', data.status);
});

eventSource.addEventListener('result', (e) => {
  const data = JSON.parse(e.data);
  console.log('Results:', data.results);
  eventSource.close();
});

eventSource.addEventListener('error', (e) => {
  const data = JSON.parse(e.data);
  console.error('Error:', data.error);
  eventSource.close();
});
```

#### Example (Python with sseclient):
```python
import json
from sseclient import SSEClient

url = f"http://localhost:8003/ocr/stream/{job_id}"
messages = SSEClient(url)

for msg in messages:
    if msg.event == 'result':
        result = json.loads(msg.data)
        print(f"Completed! Pages: {result['total_pages']}")
        break
    elif msg.event == 'status':
        status = json.loads(msg.data)
        print(f"Status: {status['status']}")
```

---

### 3. Check Job Status (Polling)

**GET** `/ocr/status/{job_id}`

Poll the current status of a job (alternative to SSE).

#### Response:
```json
{
  "job_id": "fc-01JCFZ8X...",
  "status": "running",
  "progress": null,
  "message": "Job is currently being processed"
}
```

Status values:
- `pending`: Job queued, not started
- `running`: Job is being processed
- `completed`: Job finished successfully
- `failed`: Job failed with error
- `expired`: Job results expired (>7 days old)

#### Example (curl):
```bash
curl "http://localhost:8003/ocr/status/fc-01JCFZ8X..."
```

---

### 4. Get Job Result

**GET** `/ocr/result/{job_id}`

Retrieve the full results of a completed job.

#### Parameters:
- `annotate` (query, optional): Return annotated images with bounding boxes (default: false)

#### Response (Success - 200):
```json
{
  "success": true,
  "job_id": "fc-01JCFZ8X...",
  "status": "completed",
  "filename": "document.pdf",
  "total_pages": 5,
  "dpi": 200,
  "model": "dotsocr",
  "results": {
    "page_0": {
      "status": "success",
      "data": [
        {
          "bbox": [100, 200, 500, 250],
          "category": "Title",
          "text": "Document Title"
        }
      ]
    }
  },
  "annotated_images": [],
  "message": "Job completed successfully",
  "completed_at": "2025-11-06T10:35:00.000Z"
}
```

#### Response (Still Running - 202):
```json
{
  "success": false,
  "job_id": "fc-01JCFZ8X...",
  "status": "running",
  "message": "Job is still being processed. Please try again later."
}
```

#### Response (Expired - 404):
```json
{
  "success": false,
  "job_id": "fc-01JCFZ8X...",
  "status": "expired",
  "message": "Job results have expired (>7 days old)"
}
```

#### Example (curl):
```bash
curl "http://localhost:8003/ocr/result/fc-01JCFZ8X..."
```

---

## Usage Patterns

### Pattern 1: Submit + SSE Stream (Recommended)

Best for real-time feedback in web applications.

```javascript
// 1. Submit job
const formData = new FormData();
formData.append('file', pdfFile);

const submitResponse = await fetch('/ocr/submit', {
  method: 'POST',
  body: formData
});

const { job_id } = await submitResponse.json();

// 2. Stream progress
const eventSource = new EventSource(`/ocr/stream/${job_id}`);

eventSource.addEventListener('result', (e) => {
  const data = JSON.parse(e.data);
  displayResults(data.results);
  eventSource.close();
});
```

### Pattern 2: Submit + Poll Status

Best for simple clients or background jobs.

```python
import requests
import time

# 1. Submit job
files = {'file': open('document.pdf', 'rb')}
response = requests.post('http://localhost:8003/ocr/submit', files=files)
job_id = response.json()['job_id']

# 2. Poll until complete
while True:
    status_response = requests.get(f'http://localhost:8003/ocr/status/{job_id}')
    status_data = status_response.json()
    
    if status_data['status'] == 'completed':
        # Get results
        result = requests.get(f'http://localhost:8003/ocr/result/{job_id}')
        print(result.json())
        break
    
    time.sleep(2)  # Wait 2 seconds before next poll
```

### Pattern 3: Fire and Forget + Retrieve Later

Best for batch processing or delayed retrieval.

```python
# Submit multiple jobs
job_ids = []
for pdf_file in pdf_files:
    response = requests.post('/ocr/submit', files={'file': open(pdf_file, 'rb')})
    job_ids.append(response.json()['job_id'])

# Later, retrieve results
for job_id in job_ids:
    result = requests.get(f'/ocr/result/{job_id}')
    if result.status_code == 200:
        process_result(result.json())
    elif result.status_code == 202:
        print(f"Job {job_id} still running")
```

---

## Models

### DotsOCR (`dotsocr`)
- **Features**: Layout detection, bounding boxes, structured JSON output
- **Output**: List of layout elements with categories (Title, Text, Table, etc.)
- **Best for**: Document understanding, layout analysis, structured data extraction

### LightOnOCR (`lightonocr`)
- **Features**: Markdown text extraction
- **Output**: Clean markdown text
- **Best for**: Text extraction, content parsing, markdown conversion

---

## Deployment

### 1. Deploy Modal Functions

```bash
cd backend
modal deploy visor_multi_ocr.py
```

This deploys:
- DotsOCR inference server
- LightOnOCR inference server
- `ocr_batch_pages` job function

### 2. Run FastAPI Server

```bash
python server.py
# or
uvicorn server:app --host 0.0.0.0 --port 8003 --reload
```

### 3. Test with Example Client

Open `sse_client_example.html` in your browser or serve it:

```bash
python -m http.server 8080
# Then open http://localhost:8080/sse_client_example.html
```

---

## Error Handling

### Common Errors

**503 - OCR backend not available**
```json
{
  "detail": "OCR backend not available. Deploy with: modal deploy visor_multi_ocr.py"
}
```
→ Deploy Modal functions first

**400 - Unsupported model**
```json
{
  "detail": "Unsupported model 'xyz'. Available models: dotsocr, lightonocr"
}
```
→ Use valid model name

**404 - Job expired**
```json
{
  "status": "expired",
  "message": "Job results have expired (>7 days old)"
}
```
→ Results are kept for 7 days only

---

## Performance Considerations

- **Concurrent Processing**: Each Modal container processes up to 8 pages in parallel
- **Job Retention**: Results are available for 7 days after completion
- **SSE Timeout**: SSE stream polls for up to 10 minutes, then times out (job continues running)
- **Auto-scaling**: Modal automatically scales based on workload

---

## Migration from Legacy API

### Old API (Blocking)
```python
POST /ocr
→ Wait for full processing
→ Returns complete results
```

### New API (Job Queue)
```python
POST /ocr/submit
→ Returns immediately with job_id
→ Stream progress via SSE
→ Retrieve results when ready
```

**Note**: The legacy `/ocr` endpoint is still available but deprecated.

---

## Complete Example

See `sse_client_example.html` for a complete working example with:
- File upload
- Job submission
- Real-time SSE progress monitoring
- Result display
- Error handling

---

## Support

For issues or questions:
1. Check Modal deployment: `modal app list`
2. Check Modal logs: `modal app logs visor-multi-ocr`
3. Check FastAPI logs in terminal
4. Verify job status: `GET /ocr/status/{job_id}`
