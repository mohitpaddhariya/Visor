import io
import base64
import json
import traceback
import asyncio
from datetime import datetime
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# -------------------------------------------------
# Load Modal Configuration
# -------------------------------------------------
try:
    with open("modal_config.json", "r") as f:
        MODAL_CONFIG = json.load(f)
    print("Loaded modal configuration successfully")
except FileNotFoundError:
    print("Warning: modal_config.json not found, using default configuration")
    MODAL_CONFIG = {
        "models": {
            "dotsocr": {
                "name": "DotsOCR",
                "description": "High-performance OCR model for document understanding",
                "supports_bounding_boxes": True,
                "features": ["layout_detection", "bbox_extraction", "structured_json_output"]
            },
            "lightonocr": {
                "name": "LightOnOCR",
                "description": "Efficient OCR model for markdown text extraction",
                "supports_bounding_boxes": False,
                "features": ["markdown_extraction", "text_recognition"]
            }
        },
        "default_model": "dotsocr",
        "supported_models": ["dotsocr", "lightonocr"]
    }
except Exception as e:
    print(f"Error loading modal_config.json: {e}")
    MODAL_CONFIG = {"models": {}, "default_model": "dotsocr", "supported_models": []}

# -------------------------------------------------
# Pydantic Models
# -------------------------------------------------
class PageResult(BaseModel):
    status: str  # "success" or "error"
    data: Optional[Any] = None  # List[dict] for dotsocr, str for lightonocr
    error: Optional[str] = None

    def model_dump(self, **kwargs):
        # Ensure data is JSON-serializable
        data = self.data
        if isinstance(data, list):
            data = [dict(d) for d in data]
        return super().model_dump(**kwargs) | {"data": data}


class OCRRequest(BaseModel):
    dpi: int = 200
    model: str = "dotsocr"  # "dotsocr" or "lightonocr"
    annotate: bool = False  # Return annotated images?
    output_folder: Optional[str] = None


class OCRJobSubmitResponse(BaseModel):
    """Response when submitting a new OCR job"""
    success: bool
    job_id: str
    filename: str
    total_pages: int
    model: str
    dpi: int
    message: str
    submitted_at: datetime


class OCRJobStatusResponse(BaseModel):
    """Response for job status check"""
    job_id: str
    status: str  # "pending", "running", "completed", "failed", "expired"
    progress: Optional[Dict[str, Any]] = None
    message: str


class OCRResponse(BaseModel):
    success: bool
    filename: str
    total_pages: int
    dpi: int
    model: str
    results: Dict[str, Dict[str, Any]]
    annotated_images: List[str] = []  # base64 PNGs if annotate=True
    message: str = ""
    created_at: datetime  # Timestamp when OCR was processed


# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(
    title="PDF OCR API (Multi-Model)",
    description="Configurable multi-model OCR API supporting layout analysis and text extraction",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Modal Function Reference
# -------------------------------------------------
from modal import Function as ModalFunction, FunctionCall

OCR_FUNCTION_APP = "visor-multi-ocr"
OCR_FUNCTION_NAME = "ocr_batch_pages"

try:
    ocr_batch_pages_fn = ModalFunction.from_name(OCR_FUNCTION_APP, OCR_FUNCTION_NAME)
    print(f"Connected to Modal function: {OCR_FUNCTION_APP}.{OCR_FUNCTION_NAME}")
except Exception as e:
    print(f"Could not connect to Modal function: {e}")
    print("   Run: modal deploy visor_multi_ocr.py")
    ocr_batch_pages_fn = None


# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
    """Convert PDF pages to PIL Images at given DPI."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def draw_boxes_on_image(image: Image.Image, boxes: List[dict]) -> Image.Image:
    """Draw layout boxes with category labels."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    color_map = {
        "Text": "#00FF00", "Title": "#FF0000", "Section-header": "#FF00FF",
        "Picture": "#00FFFF", "Table": "#FFFF00", "Formula": "#FFA500",
        "Caption": "#FFC0CB", "Footnote": "#A9A9A9", "List-item": "#90EE90",
        "Page-header": "#87CEEB", "Page-footer": "#DDA0DD",
    }

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()

    for box in boxes:
        bbox = box.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        cat = box.get("category", "Unknown")
        color = color_map.get(cat, "#FFFFFF")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = cat
        label_y = max(y1 - 25, 5)
        try:
            txt_bbox = draw.textbbox((x1, label_y), label, font=font)
            txt_bbox = (txt_bbox[0]-2, txt_bbox[1]-2, txt_bbox[2]+2, txt_bbox[3]+2)
            draw.rectangle(txt_bbox, fill=color)
            draw.text((x1, label_y), label, fill="black", font=font)
        except:
            draw.text((x1, label_y), label, fill=color)

    return img


# -------------------------------------------------
# API Endpoints
# -------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "PDF OCR API",
        "version": "2.1.0",
        "models": MODAL_CONFIG.get("supported_models", []),
        "default_model": MODAL_CONFIG.get("default_model", "dotsocr"),
        "modal_connected": ocr_batch_pages_fn is not None
    }


@app.get("/ocr/models")
async def get_available_models():
    """Get detailed information about available OCR models and their features."""
    return {
        "models": MODAL_CONFIG.get("models", {}),
        "default_model": MODAL_CONFIG.get("default_model", "dotsocr"),
        "supported_models": MODAL_CONFIG.get("supported_models", [])
    }


@app.post("/ocr", response_model=OCRResponse)
async def ocr_pdf(
    file: UploadFile = File(...),
    dpi: int = Query(200, ge=72, le=600),
    model: str = Query(None, description="OCR model to use"),
    annotate: bool = Query(False),
    output_folder: Optional[str] = None
) -> OCRResponse:
    """
    [DEPRECATED - Use /ocr/submit for job-based processing]
    
    Extract text/layout from PDF using configured OCR models.

    Available models:
    - dotsocr: Structured layout JSON with bbox, categories, and formatted text
    - lightonocr: Clean markdown text extraction
    """
    # Set default model from config if not specified
    if model is None:
        model = MODAL_CONFIG.get("default_model", "dotsocr")

    # Validate model
    supported_models = MODAL_CONFIG.get("supported_models", [])
    if model not in supported_models:
        raise HTTPException(
            400,
            f"Unsupported model '{model}'. Available models: {', '.join(supported_models)}"
        )

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")

    if ocr_batch_pages_fn is None:
        raise HTTPException(
            503,
            "OCR backend not available. Deploy with: modal deploy visor_multi_ocr.py"
        )

    try:
        pdf_bytes = await file.read()
        images = pdf_to_images(pdf_bytes, dpi=dpi)
        total_pages = len(images)

        # Convert to base64
        b64_images = [image_to_base64(img) for img in images]

        # Call multi-model OCR Modal function (blocking)
        job_metadata = {
            "filename": file.filename,
            "dpi": dpi,
            "total_pages": total_pages
        }
        result = ocr_batch_pages_fn.remote(b64_images, model=model, job_metadata=job_metadata)
        raw_results = result.get("results", {})

        # Optional: Annotate images (only for models that support bounding boxes)
        model_config = MODAL_CONFIG.get("models", {}).get(model, {})
        supports_bbox = model_config.get("supports_bounding_boxes", False)
        
        annotated_b64 = []
        if annotate and supports_bbox:
            for idx, img in enumerate(images):
                page_key = f"page_{idx}"
                page_res = raw_results.get(page_key, {})
                boxes = page_res.get("data", []) if page_res.get("status") == "success" else []
                annotated_img = draw_boxes_on_image(img, boxes)
                annotated_b64.append(image_to_base64(annotated_img))

        # Get model info for response message
        model_info = MODAL_CONFIG.get("models", {}).get(model, {})
        model_name = model_info.get("name", model.upper())

        return OCRResponse(
            success=True,
            filename=file.filename,
            total_pages=total_pages,
            dpi=dpi,
            model=model,
            results=raw_results,
            annotated_images=annotated_b64,
            message=f"Processed {total_pages} pages with {model_name}",
            created_at=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"OCR failed: {str(e)}")


@app.post("/ocr/submit", response_model=OCRJobSubmitResponse)
async def submit_ocr_job(
    file: UploadFile = File(...),
    dpi: int = Query(200, ge=72, le=600),
    model: str = Query(None, description="OCR model to use"),
) -> OCRJobSubmitResponse:
    """
    Submit a new OCR job to the queue and get a job ID.
    
    Use this endpoint to submit long-running OCR tasks. The job will be processed
    asynchronously and you can check its status using /ocr/status/{job_id} or
    stream updates via SSE at /ocr/stream/{job_id}.
    
    Available models:
    - dotsocr: Structured layout JSON with bbox, categories, and formatted text
    - lightonocr: Clean markdown text extraction
    """
    # Set default model from config if not specified
    if model is None:
        model = MODAL_CONFIG.get("default_model", "dotsocr")

    # Validate model
    supported_models = MODAL_CONFIG.get("supported_models", [])
    if model not in supported_models:
        raise HTTPException(
            400,
            f"Unsupported model '{model}'. Available models: {', '.join(supported_models)}"
        )

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")

    if ocr_batch_pages_fn is None:
        raise HTTPException(
            503,
            "OCR backend not available. Deploy with: modal deploy visor_multi_ocr.py"
        )

    try:
        pdf_bytes = await file.read()
        images = pdf_to_images(pdf_bytes, dpi=dpi)
        total_pages = len(images)

        # Convert to base64
        b64_images = [image_to_base64(img) for img in images]

        # Prepare job metadata
        job_metadata = {
            "filename": file.filename,
            "dpi": dpi,
            "total_pages": total_pages,
            "submitted_at": datetime.now().isoformat()
        }

        # Spawn the job asynchronously
        print(f"Spawning OCR job for {file.filename} ({total_pages} pages) with model {model}")
        call = await ocr_batch_pages_fn.spawn.aio(b64_images, model=model, job_metadata=job_metadata)
        job_id = call.object_id

        print(f"Job submitted: {job_id}")

        return OCRJobSubmitResponse(
            success=True,
            job_id=job_id,
            filename=file.filename,
            total_pages=total_pages,
            model=model,
            dpi=dpi,
            message=f"Job submitted successfully. Use /ocr/status/{job_id} to check status or /ocr/stream/{job_id} for real-time updates.",
            submitted_at=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Failed to submit OCR job: {str(e)}")


@app.get("/ocr/status/{job_id}", response_model=OCRJobStatusResponse)
async def get_job_status(job_id: str) -> OCRJobStatusResponse:
    """
    Check the status of an OCR job.
    
    Returns:
    - status: "pending" (queued), "running" (processing), "completed", "failed", or "expired"
    - progress: Current progress information if available
    """
    try:
        function_call = FunctionCall.from_id(job_id)
        
        # Try to get result with 0 timeout (non-blocking poll)
        try:
            result = await function_call.get.aio(timeout=0)
            
            # Job completed successfully
            return OCRJobStatusResponse(
                job_id=job_id,
                status="completed",
                progress={
                    "total_pages": result.get("total_pages", 0),
                    "model": result.get("model", "unknown"),
                    "metadata": result.get("metadata", {})
                },
                message="Job completed successfully"
            )
            
        except TimeoutError:
            # Job is still running or pending
            return OCRJobStatusResponse(
                job_id=job_id,
                status="running",
                progress=None,
                message="Job is currently being processed"
            )
            
    except Exception as e:
        error_msg = str(e)
        if "OutputExpiredError" in error_msg or "expired" in error_msg.lower():
            return OCRJobStatusResponse(
                job_id=job_id,
                status="expired",
                progress=None,
                message="Job results have expired (>7 days old)"
            )
        else:
            return OCRJobStatusResponse(
                job_id=job_id,
                status="failed",
                progress=None,
                message=f"Job failed: {error_msg}"
            )


@app.get("/ocr/result/{job_id}")
async def get_job_result(
    job_id: str,
    annotate: bool = Query(False, description="Return annotated images with bounding boxes")
):
    """
    Get the result of a completed OCR job.
    
    Returns the full OCR results including all pages. If the job is not yet complete,
    returns a 202 status. If expired, returns 404.
    """
    try:
        function_call = FunctionCall.from_id(job_id)
        
        try:
            result = await function_call.get.aio(timeout=0)
            
            # Job completed - return full results
            raw_results = result.get("results", {})
            metadata = result.get("metadata", {})
            
            # Handle annotation if requested
            annotated_b64 = []
            if annotate:
                model = result.get("model", "dotsocr")
                model_config = MODAL_CONFIG.get("models", {}).get(model, {})
                supports_bbox = model_config.get("supports_bounding_boxes", False)
                
                if supports_bbox and "images_b64" in metadata:
                    # Would need to re-render images with boxes
                    # For now, just indicate it's not available in stored results
                    pass
            
            return {
                "success": True,
                "job_id": job_id,
                "status": "completed",
                "filename": metadata.get("filename", "unknown"),
                "total_pages": result.get("total_pages", 0),
                "dpi": metadata.get("dpi", 200),
                "model": result.get("model", "unknown"),
                "results": raw_results,
                "annotated_images": annotated_b64,
                "message": "Job completed successfully",
                "completed_at": datetime.now()
            }
            
        except TimeoutError:
            # Job still running
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=202,
                content={
                    "success": False,
                    "job_id": job_id,
                    "status": "running",
                    "message": "Job is still being processed. Please try again later."
                }
            )
            
    except Exception as e:
        error_msg = str(e)
        if "OutputExpiredError" in error_msg or "expired" in error_msg.lower():
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "job_id": job_id,
                    "status": "expired",
                    "message": "Job results have expired (>7 days old)"
                }
            )
        else:
            raise HTTPException(500, f"Failed to retrieve job result: {error_msg}")


@app.get("/ocr/stream/{job_id}")
async def stream_job_progress(job_id: str):
    """
    Stream job progress updates via Server-Sent Events (SSE).
    
    This endpoint streams real-time updates about the job status. The client should
    listen to this event stream to get progress updates without polling.
    
    Events sent:
    - status: Current job status (pending, running, completed, failed)
    - progress: Progress updates with page counts
    - result: Final result when job completes
    - error: Error information if job fails
    """
    async def event_generator():
        try:
            function_call = FunctionCall.from_id(job_id)
            
            # Send initial status
            yield f"event: status\ndata: {json.dumps({'status': 'pending', 'job_id': job_id})}\n\n"
            await asyncio.sleep(0.1)
            
            # Poll for completion
            max_wait = 600  # 10 minutes max
            poll_interval = 2  # Poll every 2 seconds
            elapsed = 0
            
            while elapsed < max_wait:
                try:
                    result = await function_call.get.aio(timeout=0)
                    
                    # Job completed!
                    yield f"event: status\ndata: {json.dumps({'status': 'completed', 'job_id': job_id})}\n\n"
                    await asyncio.sleep(0.1)
                    
                    # Send final result
                    result_data = {
                        "job_id": job_id,
                        "status": "completed",
                        "total_pages": result.get("total_pages", 0),
                        "model": result.get("model", "unknown"),
                        "results": result.get("results", {}),
                        "metadata": result.get("metadata", {})
                    }
                    yield f"event: result\ndata: {json.dumps(result_data)}\n\n"
                    yield f"event: close\ndata: {json.dumps({'message': 'Stream complete'})}\n\n"
                    break
                    
                except TimeoutError:
                    # Still running
                    yield f"event: status\ndata: {json.dumps({'status': 'running', 'job_id': job_id, 'elapsed': elapsed})}\n\n"
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval
                    
            else:
                # Timeout reached
                yield f"event: timeout\ndata: {json.dumps({'status': 'timeout', 'job_id': job_id, 'message': 'Job still running after 10 minutes'})}\n\n"
                
        except Exception as e:
            error_msg = str(e)
            yield f"event: error\ndata: {json.dumps({'status': 'error', 'job_id': job_id, 'error': error_msg})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable proxy buffering
        }
    )


# -------------------------------------------------
# Local Dev Runner
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8003, reload=True)