# Visor Backend - Multi-Model OCR API

A high-performance PDF OCR API powered by Modal's serverless infrastructure, supporting multiple OCR models with job queue processing and real-time progress streaming.

## Features

- 🚀 **Multi-Model Support**: DotsOCR (layout detection) and LightOnOCR (markdown extraction)
- ⚡ **Job Queue System**: Asynchronous processing with Modal's job queue
- 📡 **Real-time Updates**: Server-Sent Events (SSE) for progress streaming
- 🔄 **Auto-scaling**: Modal automatically scales based on workload
- 📦 **Batch Processing**: Process multiple PDFs concurrently
- 🎯 **Layout Detection**: Extract bounding boxes, categories, and structured data
- 📝 **Markdown Export**: Clean text extraction in markdown format

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Client    │────▶│ FastAPI      │────▶│ Modal Functions │
│   (HTTP)    │◀────│ Server       │◀────│ (GPU Workers)   │
└─────────────┘     └──────────────┘     └─────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Job Queue    │
                    │ (SSE Stream) │
                    └──────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install modal fastapi uvicorn python-multipart pillow pymupdf requests
```

### 2. Deploy Modal Functions

```bash
# Login to Modal
modal token new

# Deploy the OCR functions
modal deploy visor_multi_ocr.py
```

This deploys:
- DotsOCR vLLM inference server
- LightOnOCR vLLM inference server  
- `ocr_batch_pages` job processing function

### 3. Run the FastAPI Server

```bash
python server.py
# or with auto-reload
uvicorn server:app --reload --port 8003
```

The API will be available at `http://localhost:8003`

## API Documentation

### Interactive Docs

- Swagger UI: http://localhost:8003/docs
- ReDoc: http://localhost:8003/redoc

### Quick Reference

#### 1. Submit OCR Job (Recommended)

```bash
curl -X POST "http://localhost:8003/ocr/submit?model=dotsocr&dpi=200" \
  -F "file=@document.pdf"
```

Response:
```json
{
  "job_id": "fc-01JCFZ8X...",
  "filename": "document.pdf",
  "total_pages": 5,
  "model": "dotsocr"
}
```

#### 2. Stream Progress (SSE)

```bash
curl -N "http://localhost:8003/ocr/stream/fc-01JCFZ8X..."
```

Or use the JavaScript EventSource API:
```javascript
const eventSource = new EventSource('/ocr/stream/' + jobId);
eventSource.addEventListener('result', (e) => {
  const data = JSON.parse(e.data);
  console.log('Results:', data);
});
```

#### 3. Check Job Status

```bash
curl "http://localhost:8003/ocr/status/fc-01JCFZ8X..."
```

#### 4. Get Results

```bash
curl "http://localhost:8003/ocr/result/fc-01JCFZ8X..."
```

For complete API documentation, see [JOB_QUEUE_API.md](./JOB_QUEUE_API.md)

## Usage Examples

### Python Client

```bash
# Simple polling
./client_example.py document.pdf poll dotsocr

# SSE streaming (requires: pip install sseclient-py)
./client_example.py document.pdf sse dotsocr

# Batch processing
./client_example.py doc1.pdf doc2.pdf doc3.pdf batch
```

### HTML/JavaScript Client

Open `sse_client_example.html` in your browser for a complete web interface.

### Python SDK

```python
import requests

# Submit job
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8003/ocr/submit',
        files={'file': f},
        params={'model': 'dotsocr', 'dpi': 200}
    )
    job_id = response.json()['job_id']

# Poll for result
import time
while True:
    status = requests.get(f'http://localhost:8003/ocr/status/{job_id}').json()
    if status['status'] == 'completed':
        result = requests.get(f'http://localhost:8003/ocr/result/{job_id}').json()
        print(result)
        break
    time.sleep(2)
```

## OCR Models

### DotsOCR

**Best for**: Document layout analysis, structured data extraction

**Features**:
- Layout detection (11 categories: Title, Text, Table, Formula, etc.)
- Bounding box extraction
- Structured JSON output
- LaTeX for formulas, HTML for tables, Markdown for text

**Output Example**:
```json
{
  "page_0": {
    "status": "success",
    "data": [
      {
        "bbox": [100, 200, 500, 250],
        "category": "Title",
        "text": "Document Title"
      },
      {
        "bbox": [100, 300, 500, 800],
        "category": "Text",
        "text": "Paragraph content..."
      }
    ]
  }
}
```

### LightOnOCR

**Best for**: Clean text extraction, markdown conversion

**Features**:
- Efficient markdown extraction
- Fast processing
- Clean text output

**Output Example**:
```json
{
  "page_0": {
    "status": "success",
    "data": "# Document Title\n\nParagraph content..."
  }
}
```

## Configuration

Edit `modal_config.json` to customize models and features:

```json
{
  "models": {
    "dotsocr": {
      "name": "DotsOCR",
      "supports_bounding_boxes": true,
      "features": ["layout_detection", "bbox_extraction"]
    }
  },
  "default_model": "dotsocr",
  "supported_models": ["dotsocr", "lightonocr"]
}
```

## Performance

- **Concurrent Processing**: Up to 8 pages per container in parallel
- **Auto-scaling**: Modal scales from 0 to N containers based on demand
- **GPU Support**: L40S GPUs for fast inference
- **Job Retention**: Results stored for 7 days
- **SSE Timeout**: 10 minutes (job continues running after timeout)

## Development

### Project Structure

```
backend/
├── visor_multi_ocr.py       # Modal functions (GPU workers)
├── server.py                # FastAPI server (job queue)
├── modal_config.json        # Model configuration
├── client_example.py        # Python client example
├── sse_client_example.html  # Web client example
├── JOB_QUEUE_API.md        # Complete API docs
└── evaluate.py             # Evaluation utilities
```

### Testing

```bash
# Test with a sample PDF
curl -X POST "http://localhost:8003/ocr/submit" \
  -F "file=@sample_pdfs/test.pdf" \
  -F "model=dotsocr"

# Check Modal logs
modal app logs visor-multi-ocr
```

### Debugging

```bash
# Check Modal deployment
modal app list

# View function details
modal app show visor-multi-ocr

# Stream logs
modal app logs visor-multi-ocr --follow
```

## Migration Guide

### From Legacy Blocking API

**Old** (blocking):
```python
response = requests.post('/ocr', files={'file': pdf}, params={'model': 'dotsocr'})
results = response.json()['results']  # Wait for full processing
```

**New** (job queue):
```python
# Submit job
response = requests.post('/ocr/submit', files={'file': pdf}, params={'model': 'dotsocr'})
job_id = response.json()['job_id']  # Returns immediately

# Stream progress or poll
from sseclient import SSEClient
for msg in SSEClient(f'/ocr/stream/{job_id}'):
    if msg.event == 'result':
        results = json.loads(msg.data)['results']
        break
```

## Troubleshooting

### "OCR backend not available"

→ Deploy Modal functions: `modal deploy visor_multi_ocr.py`

### "Job expired"

→ Results are kept for 7 days only. Submit a new job.

### SSE Connection Issues

→ Check CORS settings and ensure no proxy buffering

### Slow Processing

→ Check Modal logs for GPU availability: `modal app logs visor-multi-ocr`

## Resources

- [Modal Documentation](https://modal.com/docs)
- [Job Queue Guide](https://modal.com/docs/guide/job-queue)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)

## License

See project root LICENSE file.
