import io
import base64
import json
import traceback
from datetime import datetime
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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
from modal import Function as ModalFunction

OCR_FUNCTION_APP = "visor-multi-ocr"
OCR_FUNCTION_NAME = "ocr_batch_pages"

try:
    ocr_batch_pages_fn = ModalFunction.from_name(OCR_FUNCTION_APP, OCR_FUNCTION_NAME)
    print(f"Connected to Modal function: {OCR_FUNCTION_APP}.{OCR_FUNCTION_NAME}")
except Exception as e:
    print(f"Could not connect to Modal function: {e}")
    print("   Run: modal deploy vllm_server_multi.py")
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
            "OCR backend not available. Deploy with: modal deploy vllm_server_multi.py"
        )

    try:
        pdf_bytes = await file.read()
        images = pdf_to_images(pdf_bytes, dpi=dpi)
        total_pages = len(images)

        # Convert to base64
        b64_images = [image_to_base64(img) for img in images]

        # Call multi-model OCR Modal function
        raw_results = ocr_batch_pages_fn.remote(b64_images, model=model)

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


# -------------------------------------------------
# Local Dev Runner
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8003, reload=True)