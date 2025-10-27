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

SERIAL_TEST_PAGES = 5   # Number of pages to test one-by-one
BATCH_TEST_PAGES = 20   # Number of pages to test in batch mode
VLLM_BATCH_SIZE = 8     # vLLM internal batch size (matches server config)

# --- Helper Functions ---

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
    Main evaluation function for vLLM batch processing.
    
    Usage:
    modal run evaluate.py --pdf-path /path/to/your-test-doc.pdf
    """
    
    # --- 1. Setup ---
    print(f"--- 📈 Starting Visor OCR Evaluation (vLLM Batch Mode) ---")
    print(f"Using L40S Price: ${L40S_PRICE_PER_SEC:.6f}/sec (${L40S_PRICE_PER_HOUR}/hr)")
    print(f"vLLM Batch Size: {VLLM_BATCH_SIZE} pages\n")
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ Error: File not found at {pdf_path}")
        return

    print(f"Loading PDF: {pdf_file.name}")
    try:
        # Get the batch processing function
        ocr_batch_pages_fn = modal.Function.from_name("visor-vllm", "ocr_batch_pages")
        
        # Load and convert PDF
        pdf_bytes = pdf_file.read_bytes()
        images = pdf_to_images(pdf_bytes, dpi=200)
        all_pages_b64 = [image_to_base64(img) for img in images]
        total_pages = len(all_pages_b64)
        print(f"📄 Converted PDF to {total_pages} pages.")
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return

    # --- 2. Test 1: Serial Processing (Simulated Single-Page Experience) ---
    print("\n--- Test 1: Serial Processing (Processing 1 page at a time) ---")
    print(f"Testing {SERIAL_TEST_PAGES} pages individually to measure single-page latency...")
    
    serial_times = []
    serial_test_subset = all_pages_b64[:min(SERIAL_TEST_PAGES, total_pages)]

    if not serial_test_subset:
        print("⚠️ Skipped serial test: No pages to test.")
    else:
        for i, page_b64 in enumerate(serial_test_subset):
            print(f"  Processing page {i+1}/{len(serial_test_subset)}...", end="", flush=True)
            start_time = time.monotonic()
            try:
                # Call batch function with single page
                result = await ocr_batch_pages_fn.remote.aio([page_b64])
                end_time = time.monotonic()
                duration = end_time - start_time
                serial_times.append(duration)
                
                # Check if successful
                page_result = result.get('page_0', {})
                status = page_result.get('status', 'unknown')
                print(f" Done ({duration:.2f}s) - Status: {status}")
            except Exception as e:
                print(f" Error: {e}")
        
        if serial_times:
            avg_serial_time = np.mean(serial_times)
            std_serial_time = np.std(serial_times)
            min_serial_time = np.min(serial_times)
            max_serial_time = np.max(serial_times)
            
            print(f"\n📊 **Serial Processing Results:**")
            print(f"   Average: {avg_serial_time:.2f}s (±{std_serial_time:.2f}s)")
            print(f"   Range: {min_serial_time:.2f}s - {max_serial_time:.2f}s")
            print(f"   (This is the time to process a single page)")

    # --- 3. Test 2: Batch Processing (vLLM Native Batching) ---
    print(f"\n--- Test 2: Batch Processing (vLLM Native Batching) ---")
    
    batch_test_subset = all_pages_b64[:min(BATCH_TEST_PAGES, total_pages)]
    
    if not batch_test_subset:
        print("⚠️ Skipped batch test: No pages to test.")
    else:
        num_test_pages = len(batch_test_subset)
        print(f"Processing {num_test_pages} pages in a single batch call...")
        print(f"(vLLM will internally batch them in groups of {VLLM_BATCH_SIZE})\n")
        
        start_batch_time = time.monotonic()
        
        try:
            # Single call to batch function with all pages
            results = await ocr_batch_pages_fn.remote.aio(batch_test_subset)
            end_batch_time = time.monotonic()
            
            total_batch_time = end_batch_time - start_batch_time
            
            # Count successes and failures
            success_count = 0
            fail_count = 0
            for page_key, page_result in results.items():
                if page_result.get('status') == 'success':
                    success_count += 1
                else:
                    fail_count += 1
            
            print(f"🏁 Finished processing {success_count} pages (with {fail_count} failures) in {total_batch_time:.2f}s.")
            
            if success_count > 0:
                # --- Calculate Metrics ---
                
                # 1. Throughput (Pages Per Second)
                throughput_pps = success_count / total_batch_time
                
                # 2. Cost Per Page (based on batch throughput)
                cost_per_page_batch = L40S_PRICE_PER_SEC / throughput_pps
                
                # 3. Effective Time Per Page (average in batch)
                effective_time_per_page = total_batch_time / success_count
                
                # 4. Speedup from vLLM batching
                num_vllm_batches = np.ceil(success_count / VLLM_BATCH_SIZE)
                avg_time_per_vllm_batch = total_batch_time / num_vllm_batches
                
                print(f"\n📊 **Batch Processing Results:**")
                print(f"   Total Throughput: {throughput_pps:.2f} pages/sec")
                print(f"   Effective Time Per Page: {effective_time_per_page:.2f}s")
                print(f"   Number of vLLM Batches: {num_vllm_batches:.0f} (of {VLLM_BATCH_SIZE} pages each)")
                print(f"   Avg Time Per vLLM Batch: {avg_time_per_vllm_batch:.2f}s")
                print(f"\n💰 **Cost Per Page (Batch): ${cost_per_page_batch:.6f}**")
                print(f"   (Approximately {1/cost_per_page_batch:.1f} pages per cent)")
                
        except Exception as e:
            print(f"❌ Error during batch processing: {e}")
            success_count = 0

    # --- 4. Summary & Comparison ---
    print("\n" + "="*70)
    print("--- 📝 FINAL SUMMARY ---")
    print("="*70)
    
    if serial_times and success_count > 0:
        # Serial metrics
        serial_cost_per_page = avg_serial_time * L40S_PRICE_PER_SEC
        
        print(f"\n📌 **Processing Mode Comparison:**\n")
        
        print(f"┌─ SERIAL (1 page at a time) ─────────────────────────────┐")
        print(f"│ Time per page:  {avg_serial_time:.2f}s                           ")
        print(f"│ Cost per page:  ${serial_cost_per_page:.6f}                      ")
        print(f"│ Throughput:     {1/avg_serial_time:.2f} pages/sec                ")
        print(f"└──────────────────────────────────────────────────────────┘")
        
        print(f"\n┌─ BATCH (vLLM native batching of {VLLM_BATCH_SIZE}) ───────────────────┐")
        print(f"│ Time per page:  {effective_time_per_page:.2f}s (effective)              ")
        print(f"│ Cost per page:  ${cost_per_page_batch:.6f}                      ")
        print(f"│ Throughput:     {throughput_pps:.2f} pages/sec                ")
        print(f"└──────────────────────────────────────────────────────────┘")
        
        # Calculate improvements
        speedup = avg_serial_time / effective_time_per_page
        cost_savings = (serial_cost_per_page - cost_per_page_batch) / serial_cost_per_page
        
        print(f"\n✨ **Performance Improvements:**")
        print(f"   🚀 Speedup: {speedup:.2f}x faster")
        print(f"   💰 Cost Reduction: {cost_savings:.1%} cheaper")
        print(f"   📈 Throughput Increase: {throughput_pps / (1/avg_serial_time):.2f}x")
        
        # Projections for large documents
        print(f"\n📊 **Projections for a 100-page document:**")
        serial_100_time = avg_serial_time * 100
        serial_100_cost = serial_cost_per_page * 100
        batch_100_time = effective_time_per_page * 100
        batch_100_cost = cost_per_page_batch * 100
        
        print(f"   Serial:  {serial_100_time/60:.1f} minutes, ${serial_100_cost:.3f}")
        print(f"   Batch:   {batch_100_time/60:.1f} minutes, ${batch_100_cost:.3f}")
        print(f"   Savings: {(serial_100_time - batch_100_time)/60:.1f} minutes, ${serial_100_cost - batch_100_cost:.3f}")
        
    elif serial_times:
        print("\n⚠️ Batch processing failed. Only serial results available.")
        print(f"   Average time per page: {avg_serial_time:.2f}s")
        print(f"   Cost per page: ${avg_serial_time * L40S_PRICE_PER_SEC:.6f}")
    elif success_count > 0:
        print("\n⚠️ Serial processing skipped. Only batch results available.")
        print(f"   Effective time per page: {effective_time_per_page:.2f}s")
        print(f"   Cost per page: ${cost_per_page_batch:.6f}")
    else:
        print("\n❌ No successful processing. Please check your setup and try again.")
    
    print("\n" + "="*70)