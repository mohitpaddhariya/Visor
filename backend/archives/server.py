import io
import base64
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json

# -------------------------------------------------
# Pydantic Models
# -------------------------------------------------

class PageResult(BaseModel):
    """Holds the result for a single page."""
    status: str  # "success" or "error"
    data: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

class OCRRequest(BaseModel):
    """Request model for OCR endpoint"""
    dpi: int = 200
    output_folder: Optional[str] = None

class OCRResponse(BaseModel):
    """The main API response."""
    success: bool
    filename: str
    total_pages: int
    dpi: int
    results: Dict[str, Dict[str, Any]]  # Raw vLLM response
    message: str = ""

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(
    title="PDF OCR API",
    description="Extract text and layout from PDF documents using DotsOCR with vLLM batch processing",
    version="2.0.0"
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

try:
    ocr_batch_pages_fn = ModalFunction.from_name("visor-vllm", "ocr_batch_pages")
except Exception as e:
    print(f"⚠️  ERROR: Could not find Modal function 'ocr_batch_pages'.")
    print(f"   Make sure to deploy first: modal deploy visor_multi_ocr.py")
    print(f"   Error: {e}")
    ocr_batch_pages_fn = None

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
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    print(f"Drawing {len(boxes)} boxes on image...")
    
    color_map = {
        "Text": "#00FF00",
        "Title": "#FF0000",
        "Section-header": "#FF00FF",
        "Picture": "#00FFFF",
        "Table": "#FFFF00",
        "Formula": "#FFA500",
        "Caption": "#FFC0CB",
        "Footnote": "#A9A9A9",
        "List-item": "#90EE90",
        "Page-header": "#87CEEB",
        "Page-footer": "#DDA0DD",
    }
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
    
    for i, box in enumerate(boxes):
        bbox = box.get("bbox", [])
        if len(bbox) != 4:
            print(f"Skipping box {i}: invalid bbox {bbox}")
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        category = box.get("category", "Unknown")
        color = color_map.get(category, "#FFFFFF")
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        label = f"{category}"
        label_y = max(y1 - 25, 5)
        
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
# API Endpoints
# -------------------------------------------------
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "PDF OCR API with vLLM Batch Processing",
        "version": "2.0.0",
        "modal_connected": ocr_batch_pages_fn is not None
    }

@app.post("/ocr", response_model=OCRResponse)
async def ocr_pdf_save(
    file: UploadFile = File(...),
    dpi: int = 200,
    output_folder: Optional[str] = None
) -> OCRResponse:
    """
    Extract text and layout from a PDF using vLLM batch processing.
    Returns the raw JSON response from vLLM.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if ocr_batch_pages_fn is None:
        raise HTTPException(
            status_code=503,
            detail="Modal OCR function not available. Deploy with: modal deploy visor_multi_ocr.py"
        )
    
    try:
        # # Create output folder
        # if output_folder is None:
        #     base_name = Path(file.filename).stem
        #     output_folder = f"./ocr_output/{base_name}"
        
        # output_path = Path(output_folder)
        # output_path.mkdir(parents=True, exist_ok=True)
        # print(f"Saving results to: {output_path}")
        
        pdf_bytes = await file.read()
        
        print(f"Converting PDF to images (DPI: {dpi})...")
        images = pdf_to_images(pdf_bytes, dpi=dpi)
        total_pages = len(images)
        print(f"Converted {total_pages} pages")
        
        print("Encoding images to base64...")
        image_b64_list = [image_to_base64(img) for img in images]
        
        # Call the batch processing function (single Modal call for all pages)
        print(f"Calling Modal OCR batch function for {len(image_b64_list)} pages...")
        results_dict = ocr_batch_pages_fn.remote(image_b64_list)
        
        # Return the raw vLLM response
        return OCRResponse(
            success=True,
            filename=file.filename,
            total_pages=total_pages,
            dpi=dpi,
            results=results_dict,  # Raw JSON from vLLM
            message=f"Successfully processed {total_pages} pages using vLLM batch processing"
        )
        
        # # OLD LOGIC - Convert to PageResult objects and save files
        # # Convert to PageResult objects
        # results_by_page = {}
        # for page_key, result_data in results_dict.items():
        #     results_by_page[page_key] = PageResult(**result_data)
        
        # saved_files = {
        #     "json_file": None,
        #     "annotated_images": [],
        #     "original_images": []
        # }
        
        # # Save JSON results
        # json_path = output_path / "ocr_results.json"
        # serializable_results = {key: value.model_dump() for key, value in results_by_page.items()}
        
        # with open(json_path, 'w') as f:
        #     json.dump({
        #         "filename": file.filename,
        #         "total_pages": total_pages,
        #         "dpi": dpi,
        #         "results_by_page": serializable_results
        #     }, f, indent=2)
        # saved_files["json_file"] = str(json_path)
        # print(f"Saved JSON to: {json_path}")
        
        # # Save images with annotations
        # for page_num, img in enumerate(images):
        #     # Save original image
        #     orig_path = output_path / f"page_{page_num}_original.png"
        #     img.save(orig_path)
        #     saved_files["original_images"].append(str(orig_path))
            
        #     # Get boxes for annotation
        #     page_key = f"page_{page_num}"
        #     page_result = results_by_page.get(page_key)
            
        #     boxes = []
        #     if page_result and page_result.status == "success":
        #         boxes = page_result.data
        #     elif page_result and page_result.status == "error":
        #         print(f"Skipping annotations for page {page_num} due to error: {page_result.error}")
            
        #     annotated_img = draw_boxes_on_image(img, boxes)
        #     annotated_path = output_path / f"page_{page_num}_annotated.png"
        #     annotated_img.save(annotated_path)
        #     saved_files["annotated_images"].append(str(annotated_path))
            
        #     print(f"Saved page {page_num}: original and annotated")
        
        # return {
        #     "success": True,
        #     "output_folder": str(output_path),
        #     "total_pages": total_pages,
        #     "files": saved_files,
        #     "message": f"Successfully processed {total_pages} pages using vLLM batch processing"
        # }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8003, reload=True)