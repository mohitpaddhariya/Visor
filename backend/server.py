import io
import base64
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import modal
from pathlib import Path
import json

# -------------------------------------------------
# Pydantic Models
# -------------------------------------------------

class PageResult(BaseModel):
    """Holds the result for a single page, which can be success or error."""
    status: str  # "success" or "error"
    data: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

class OCRResponse(BaseModel):
    """The main API response, now contains a dictionary of PageResult objects."""
    success: bool
    results: Dict[str, PageResult]  # e.g., {"page_0": PageResult, "page_1": PageResult}
    total_pages: int
    message: str = ""

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(
    title="PDF OCR API",
    description="Extract text and layout from PDF documents using DotsOCR",
    version="1.0.0"
)

# CORS configuration
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

# Make sure to deploy first: modal deploy vllm_server.py
try:
    ocr_single_page_fn = ModalFunction.from_name("visor-vllm", "ocr_single_page")
except Exception as e:
    print(f"⚠️  ERROR: Could not find Modal function 'ocr_single_page'.")
    print(f"   Make sure to deploy first: modal deploy vllm_server.py")
    print(f"   Error: {e}")
    ocr_single_page_fn = None

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
    """Convert PDF pages to PIL Images"""
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    pdf_document.close()
    return images

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URL"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"

def draw_boxes_on_image(image: Image.Image, boxes: List[dict]) -> Image.Image:
    """Draw bounding boxes on an image with category labels"""
    # Create a copy to avoid modifying the original
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    print(f"Drawing {len(boxes)} boxes on image...")
    
    # Color mapping for different categories
    color_map = {
        "Text": "#00FF00",          # Green
        "Title": "#FF0000",         # Red
        "Section-header": "#FF00FF", # Magenta
        "Picture": "#00FFFF",       # Cyan
        "Table": "#FFFF00",         # Yellow
        "Formula": "#FFA500",       # Orange
        "Caption": "#FFC0CB",       # Pink
        "Footnote": "#A9A9A9",      # Gray
        "List-item": "#90EE90",     # Light Green
        "Page-header": "#87CEEB",   # Sky Blue
        "Page-footer": "#DDA0DD",   # Plum
    }
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        try:
            # Try alternative font path
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
    
    # Draw each box
    for i, box in enumerate(boxes):
        bbox = box.get("bbox", [])
        if len(bbox) != 4:
            print(f"Skipping box {i}: invalid bbox {bbox}")
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        category = box.get("category", "Unknown")
        
        # Get color for this category
        color = color_map.get(category, "#FFFFFF")
        
        # Draw rectangle with thicker border
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Draw category label with background
        label = f"{category}"
        label_y = max(y1 - 25, 5)  # Keep it within image bounds
        
        try:
            bbox_text = draw.textbbox((x1, label_y), label, font=font)
            bbox_text = (bbox_text[0] - 2, bbox_text[1] - 2, bbox_text[2] + 2, bbox_text[3] + 2)
            draw.rectangle(bbox_text, fill=color)
            draw.text((x1, label_y), label, fill="black", font=font)
        except:
            draw.text((x1, label_y), label, fill=color, font=font)
    
    print(f"Finished drawing boxes")
    return img_copy

# -------------------------------------------------
# Internal Helper to run and process map
# -------------------------------------------------
async def run_ocr_map(image_b64_list: List[str]) -> Dict[str, PageResult]:
    """
    Runs the .map() call with exception handling and returns the
    structured dictionary.
    """
    if ocr_single_page_fn is None:
        raise HTTPException(
            status_code=503,
            detail="Modal OCR function not available. Deploy with: modal deploy vllm_server.py"
        )
        
    print(f"Calling Modal OCR function via .map() for {len(image_b64_list)} pages...")
    
    results_dict: Dict[str, PageResult] = {}
    
    # We use .map.aio() for async compatibility
    i = 0
    try:
        async for result in ocr_single_page_fn.map.aio(image_b64_list, return_exceptions=True):
            page_key = f"page_{i}"
            if isinstance(result, Exception):
                # If the result is an exception, log it as an error
                print(f"Error on page {i}: {result}")
                results_dict[page_key] = PageResult(
                    status="error",
                    error=str(result)
                )
            else:
                # Otherwise, it's a successful result
                results_dict[page_key] = PageResult(
                    status="success",
                    data=result
                )
            i += 1
            
    except Exception as e:
        # This catches errors with the map call itself (e.g., connection issue)
        print(f"A critical error occurred during the .map() call: {e}")
        # Add error entries for any pages that didn't get processed
        while i < len(image_b64_list):
            results_dict[f"page_{i}"] = PageResult(status="error", error=f"Map call failed: {e}")
            i += 1
            
    return results_dict

# -------------------------------------------------
# API Endpoints (UPDATED)
# -------------------------------------------------
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "PDF OCR API",
        "version": "1.0.0",
        "modal_connected": ocr_single_page_fn is not None
    }

@app.post("/ocr")
async def ocr_pdf_save(
    file: UploadFile = File(...),
    dpi: int = 200,
    output_folder: Optional[str] = None
):
    """
    Extract text and layout from a PDF and save results to a folder
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Create output folder
        if output_folder is None:
            base_name = Path(file.filename).stem
            output_folder = f"./ocr_output/{base_name}"
        
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to: {output_path}")
        
        pdf_bytes = await file.read()
        
        print(f"Converting PDF to images (DPI: {dpi})...")
        images = pdf_to_images(pdf_bytes, dpi=dpi)
        total_pages = len(images)
        print(f"Converted {total_pages} pages")
        
        print("Encoding images to base64...")
        image_b64_list = [image_to_base64(img) for img in images]
        
        # Run the robust map call
        results_by_page_dict = await run_ocr_map(image_b64_list)

        saved_files = {
            "json_file": None,
            "annotated_images": [],
            "original_images": []
        }
        
        # Save JSON results
        json_path = output_path / "ocr_results.json"
        
        # We need to convert Pydantic models to dicts for json.dump
        serializable_results = {key: value.model_dump() for key, value in results_by_page_dict.items()}
        
        with open(json_path, 'w') as f:
            json.dump({
                "filename": file.filename,
                "total_pages": total_pages,
                "dpi": dpi,
                "results_by_page": serializable_results # Save the new dict format
            }, f, indent=2)
        saved_files["json_file"] = str(json_path)
        print(f"Saved JSON to: {json_path}")
        
        # Save images with annotations
        for page_num, img in enumerate(images):
            # Save original image
            orig_path = output_path / f"page_{page_num}_original.png"
            img.save(orig_path)
            saved_files["original_images"].append(str(orig_path))
            
            # Check status before trying to get data for boxes
            page_key = f"page_{page_num}"
            page_result = results_by_page_dict.get(page_key)
            
            boxes = []
            if page_result and page_result.status == "success":
                boxes = page_result.data
            elif page_result and page_result.status == "error":
                print(f"Skipping annotations for page {page_num} due to error: {page_result.error}")
            
            annotated_img = draw_boxes_on_image(img, boxes)
            annotated_path = output_path / f"page_{page_num}_annotated.png"
            annotated_img.save(annotated_path)
            saved_files["annotated_images"].append(str(annotated_path))
            
            print(f"Saved page {page_num}: original and annotated")
        
        return {
            "success": True,
            "output_folder": str(output_path),
            "total_pages": total_pages,
            "files": saved_files,
            "message": f"Successfully saved {total_pages} pages to {output_path}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing PDF save: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# -------------------------------------------------
# Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8080
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)