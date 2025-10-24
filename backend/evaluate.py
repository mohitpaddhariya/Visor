import modal
import time
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import io
import base64
import argparse
from pathlib import Path
from typing import List

# --- Configuration ---
# Price for L40S GPU on Modal, as of October 2025
# Source: https://modal.com/pricing
L40S_PRICE_PER_HOUR = 1.95
L40S_PRICE_PER_SEC = L40S_PRICE_PER_HOUR / 3600  # Approx $0.000542

SERIAL_TEST_PAGES = 5  # Number of pages to test one-by-one
PARALLEL_TEST_PAGES = 20 # Number of pages to test all at once

# --- Helper Functions (copied from fastapi_server.py) ---

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

# --- Modal App ---

app = modal.App()

@app.local_entrypoint()
async def main(pdf_path: str):
    """
    Main evaluation function.
    
    Usage:
    modal run evaluate.py --pdf-path /path/to/your-test-doc.pdf
    """
    
    # --- 1. Setup ---
    print(f"--- 📈 Starting Visor OCR Evaluation ---")
    print(f"Using L40S Price: ${L40S_PRICE_PER_SEC:.6f}/sec (${L40S_PRICE_PER_HOUR}/hr)")
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ Error: File not found at {pdf_path}")
        return

    print(f"Loading PDF: {pdf_file.name}")
    try:
        # Get the remote function
        ocr_single_page_fn = modal.Function.from_name("visor-vllm", "ocr_single_page")
        
        # Load and convert PDF
        pdf_bytes = pdf_file.read_bytes()
        images = pdf_to_images(pdf_bytes, dpi=200)
        all_pages_b64 = [image_to_base64(img) for img in images]
        total_pages = len(all_pages_b64)
        print(f"📄 Converted PDF to {total_pages} pages.")
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return

    # --- 2. Test 1: Serial Latency (User Experience) ---
    print("\n--- Test 1: Serial Latency (Time per single page) ---")
    print(f"Running {SERIAL_TEST_PAGES} pages one by one...")
    
    serial_times = []
    serial_test_subset = all_pages_b64[:min(SERIAL_TEST_PAGES, total_pages)]

    if not serial_test_subset:
        print("⚠️ Skipped serial test: No pages to test.")
    else:
        for i, page_b64 in enumerate(serial_test_subset):
            print(f"  Testing page {i+1}/{len(serial_test_subset)}...", end="", flush=True)
            start_time = time.monotonic()
            try:
                await ocr_single_page_fn.remote.aio(page_b64)
                end_time = time.monotonic()
                duration = end_time - start_time
                serial_times.append(duration)
                print(f" Done ({duration:.2f}s)")
            except Exception as e:
                print(f" Error on page {i+1}: {e}")
        
        if serial_times:
            avg_serial_time = np.mean(serial_times)
            std_serial_time = np.std(serial_times)
            print(f"\n📊 **Average Latency per Page: {avg_serial_time:.2f}s** (±{std_serial_time:.2f}s)")
            print(f"   (This is the 'wall clock' time a user waits for one page)")

    # --- 3. Test 2: Parallel Throughput & Cost (Full Document) ---
    print(f"\n--- Test 2: Parallel Throughput & Cost (with {PARALLEL_TEST_PAGES}-page batch) ---")
    
    parallel_test_subset = all_pages_b64[:min(PARALLEL_TEST_PAGES, total_pages)]
    
    if not parallel_test_subset:
        print("⚠️ Skipped parallel test: No pages to test.")
    else:
        num_test_pages = len(parallel_test_subset)
        print(f"Running {num_test_pages} pages in parallel (up to {PARALLEL_TEST_PAGES} at a time)...")
        
        start_parallel_time = time.monotonic()
        
        # Run all pages in parallel, collecting exceptions
        results = []
        async for result in ocr_single_page_fn.map.aio(parallel_test_subset, return_exceptions=True):
            results.append(result)
            
        end_parallel_time = time.monotonic()
        
        total_parallel_time = end_parallel_time - start_parallel_time
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        fail_count = len(results) - success_count
        
        print(f"\n🏁 Finished processing {success_count} pages (with {fail_count} failures) in {total_parallel_time:.2f}s.")
        
        if success_count > 0:
            # --- Calculate Metrics ---
            
            # 1. Throughput (Pages Per Second)
            throughput_pps = success_count / total_parallel_time
            
            # 2. Cost Per Page (based on throughput)
            # Cost per page = (Cost per second) / (Pages per second)
            cost_per_page = L40S_PRICE_PER_SEC / throughput_pps
            
            # 3. Effective Time Per Page (the average time in a parallel batch)
            # This is different from serial time!
            effective_time_per_page = total_parallel_time / success_count
            
            print(f"📊 **Total Throughput: {throughput_pps:.2f} pages/sec**")
            print(f"📊 **Effective Time Per Page (Parallel): {effective_time_per_page:.2f}s**")
            print(f"💰 **Calculated Cost Per Page: ${cost_per_page:.6f}** (approx. {1/cost_per_page:.1f} pages per cent)")

    # --- 4. Summary ---
    print("\n--- 📝 Summary ---")
    if serial_times and success_count > 0:
        print(f"**Latency (1 page):** {avg_serial_time:.2f}s")
        print(f"   *This is the time a single user waits for one page.*\n")
        print(f"**Throughput Avg. (Batch):** {effective_time_per_page:.2f}s / page")
        print(f"   *This is the effective average time per page when the system is busy.*\n")
        
        serial_cost = avg_serial_time * L40S_PRICE_PER_SEC
        print(f"**Cost (Processing 1 page at a time):** ${serial_cost:.6f} / page")
        print(f"**Cost (Processing 8 pages at once):** ${cost_per_page:.6f} / page")
        
        savings = (serial_cost - cost_per_page) / serial_cost
        print(f"\n✨ **Parallel processing is ~{savings:.0%} cheaper!**")
        
    else:
        print("Could not generate a full summary. Please re-run with a larger PDF.")