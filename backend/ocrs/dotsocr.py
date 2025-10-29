import modal
import os
import base64
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
# Using DotsOCR - A high-performance OCR model for document understanding
# Requires custom integration with vLLM via model registration
MODEL_NAME = "rednote-hilab/dots.ocr"
MODEL_SAVE_DIR = "DotsOCR"  # Clean directory name without special chars
NUMBER_OF_GPUS = 1
GPU = "L40S"
GPU_CONFIG = os.environ.get("GPU_CONFIG", f"{GPU}:{NUMBER_OF_GPUS}")
VLLM_PORT = 8000
HF_SECRET_NAME = "huggingface-secret"  # Your HF token
FAST_BOOT = False  # Set to True for faster cold starts (skips compilation)
CONCURRENT_LIMIT = 8 # <-- This is your "batch of 8"

# -------------------------------------------------
# CUDA Image Configuration (vLLM v0.9.1 requirements)
# -------------------------------------------------
cuda_version = "12.8.1"
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev", "git")
    .run_commands("pip install uv")
    .run_commands("uv pip install vllm==0.9.1 --system")
    .uv_pip_install(
        "torch==2.7.0",
        "torchvision==0.22.0",
        "torchaudio==2.7.0",
        "transformers==4.51.3",
        "pillow",
        "pymupdf",
        "huggingface_hub[hf_transfer]",
        "requests",
        "numpy",
        "qwen-vl-utils",  # Required for vision-language models
    )
    .run_commands("uv pip install flash-attn==2.8.0.post2 --no-build-isolation --system")
    .run_commands(
        "git clone https://github.com/rednote-hilab/dots.ocr.git /tmp/dots_ocr && "
        "cd /tmp/dots_ocr && "
        "uv pip install -e . --system"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/data/.cache"
    })
    .run_commands("python -c 'import torch; print(torch.__version__);'")
)

# -------------------------------------------------
# Persistent Volume for Model Caching
# -------------------------------------------------
volume = modal.Volume.from_name("visor-model-cache", create_if_missing=True)

app = modal.App("visor-vllm", image=image)

# -------------------------------------------------
# OCR Prompt
# -------------------------------------------------
OCR_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object (which is a list of the layout elements)."""

# -------------------------------------------------
# vLLM Web Server (OpenAI-compatible)
# -------------------------------------------------
@app.function(
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={"/data": volume},
    scaledown_window=3 * 60,  # Keep container alive for 3 minutes after last request
    max_containers=2,  # Max parallel containers
    timeout=60 * 60,  # 1 hour timeout
)
@modal.concurrent(max_inputs=CONCURRENT_LIMIT)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    import subprocess
    
    print(f"Starting vLLM server with concurrency limit: {CONCURRENT_LIMIT}")
    
    # Get HF token from Modal secret (your secret uses HF_TOKEN as the env var name)
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise ValueError("HuggingFace token not found.")
    
    # Model cache path
    model_path = f"/data/.cache/{MODEL_SAVE_DIR}"
    
    # Download model if not already cached
    if not os.path.exists(model_path):
        print(f"Model not found in cache. Downloading to {model_path}...")
        download_cmd = [
            "python3",
            "-c",
            f"""
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='{MODEL_NAME}',
    local_dir='{model_path}',
    token='{hf_token}'
)
"""
        ]
        subprocess.run(download_cmd, check=True)
        print(f"Model downloaded successfully to {model_path}")
        volume.commit()  # Persist the downloaded model
    else:
        print(f"Using cached model from {model_path}")
    
    # Set PYTHONPATH to include the parent directory of the model
    os.environ["PYTHONPATH"] = f"/data/.cache:{os.environ.get('PYTHONPATH', '')}"
    
    # Register DotsOCR model with vLLM by patching the vllm entrypoint
    print("Registering DotsOCR model with vLLM...")
    patch_cmd = f"""
sed -i '/^from vllm\\.entrypoints\\.cli\\.main import main$/a\\
from {MODEL_SAVE_DIR} import modeling_dots_ocr_vllm' $(which vllm)
"""
    subprocess.run(patch_cmd, shell=True, check=True)
    
    # Build vLLM server command
    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        model_path,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
    ]
    
    # Configuration options
    if FAST_BOOT:
        cmd += ["--enforce-eager"]  # Skip CUDA graph compilation for faster startup
    else:
        cmd += ["--no-enforce-eager"]
    
    cmd += [
        "--tensor-parallel-size", str(NUMBER_OF_GPUS),
        "--gpu-memory-utilization", "0.8",  # Use 80% of GPU memory (DotsOCR recommendation)
        "--chat-template-content-format", "string",
        "--trust-remote-code",
        "--served-model-name", "dotsocr-model",
        "--enable-chunked-prefill",  # Enable batch processing optimization
        "--max-num-batched-tokens", "8192",  # Batch multiple requests together
    ]
    
    print("Starting vLLM server with command:")
    print(" ".join(cmd))
    subprocess.Popen(cmd)


# -------------------------------------------------
# Batch OCR Function (processes all pages in one call)
# -------------------------------------------------
@app.function(
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=10 * 60,  # 10 minute timeout for batch processing
)
def ocr_batch_pages(image_b64_list: List[str]) -> Dict[str, Dict]:
    """
    Input: List of base64 PNG strings (data:image/png;base64,...)
    Output: Dictionary mapping page indices to their OCR results.
    This version uses parallel threads for true batch OCR processing.
    """
    try:
        server_url = serve.get_web_url()
    except modal.exception.NotFoundError:
        raise RuntimeError(
            "vLLM server is not deployed. Deploy it first with `modal deploy visor_multi_ocr.py`"
        )

    if not server_url:
        raise RuntimeError("vLLM server is not running.")

    url = f"{server_url}/v1/chat/completions"
    results = {}

    print(f"Processing {len(image_b64_list)} pages in parallel...")

    def process_page(page_idx: int, image_b64: str) -> Dict:
        """Process a single page via vLLM OCR."""
        content = [
            {"type": "text", "text": OCR_PROMPT},
            {"type": "image_url", "image_url": {"url": image_b64}},
        ]

        payload = {
            "model": "dotsocr-model",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 8192,
            "temperature": 0.0,
        }

        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"]

            outputs = json.loads(result)
            if not isinstance(outputs, list):
                raise ValueError("Model did not return a JSON list.")

            return {
                "status": "success",
                "data": outputs,
            }

        except Exception as e:
            print(f"Error processing page {page_idx}: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    # Control concurrency
    MAX_WORKERS = min(len(image_b64_list), CONCURRENT_LIMIT) # Limit to CONCURRENT_LIMIT
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(process_page, i, img): i
            for i, img in enumerate(image_b64_list)
        }

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[f"page_{idx}"] = future.result()
            except Exception as e:
                results[f"page_{idx}"] = {
                    "status": "error",
                    "error": str(e),
                }

    print("All pages processed.")
    return results