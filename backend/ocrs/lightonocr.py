import modal
import os
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
import base64
import io
import pypdfium2 as pdfium
import sys

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_NAME = "lightonai/LightOnOCR-1B-1025"
MODEL_SAVE_DIR = "LightOnOCR-1B-1025"
NUMBER_OF_GPUS = 1
GPU = "L40S"
GPU_CONFIG = os.environ.get("GPU_CONFIG", f"{GPU}:{NUMBER_OF_GPUS}")
VLLM_PORT = 8000
HF_SECRET_NAME = "huggingface-secret"
FAST_BOOT = False
CONCURRENT_LIMIT = 8

# -------------------------------------------------
# Image Build – LATEST NIGHTLY vLLM (for --async-scheduling support)
# -------------------------------------------------
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev", "git", "build-essential")
    .run_commands("pip install uv")
    .run_commands(
        "uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 "
        "--index-url https://download.pytorch.org/whl/cu124 --system"
    )
    .run_commands(
        "uv pip install vllm --system "
        "--extra-index-url https://wheels.vllm.ai/nightly "
        "--prerelease=allow"
    )
    .run_commands(
        "uv pip install transformers pillow pymupdf huggingface_hub[hf_transfer] requests numpy pypdfium2 --system"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/data/.cache"})
)
volume = modal.Volume.from_name("visor-model-cache", create_if_missing=True)
app = modal.App("lightonocr-vllm", image=image)

# -------------------------------------------------
# vLLM Server – EXACTLY as in official docs
# -------------------------------------------------
@app.function(
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={"/data": volume},
    scaledown_window=3 * 60,
    max_containers=2,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=CONCURRENT_LIMIT)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    import subprocess

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN missing")

    model_path = f"/data/.cache/{MODEL_SAVE_DIR}"
    if not os.path.exists(model_path):
        print(f"Downloading {MODEL_NAME}...")
        subprocess.run([
            "python", "-c", f"""
from huggingface_hub import snapshot_download
snapshot_download(repo_id='{MODEL_NAME}', local_dir='{model_path}', token='{hf_token}')
"""
        ], check=True)
        volume.commit()
    else:
        print(f"Using cached model: {model_path}")

    cmd = [
        "vllm", "serve", model_path,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--limit-mm-per-prompt", '{"image": 1}',
        "--async-scheduling",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.8",
        "--trust-remote-code",
        "--served-model-name", "lightonocr",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", "8192",
    ]
    if FAST_BOOT:
        cmd += ["--enforce-eager"]

    print("Starting vLLM:", " ".join(cmd))
    subprocess.Popen(cmd)


# -------------------------------------------------
# Batch OCR – NO PROMPT, JUST IMAGE
# -------------------------------------------------
@app.function(
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=10 * 60,
)
def ocr_batch_pages(image_b64_list: List[str]) -> Dict[str, Dict]:
    """
    Input: List of base64 PNG strings → data:image/png;base64,...
    Output: { "page_0": "markdown + HTML + LaTeX...", ... }
    """
    try:
        server_url = serve.get_web_url()
    except:
        raise RuntimeError("Deploy with `modal deploy` first")

    url = f"{server_url}/v1/chat/completions"
    results = {}

    def process_page(idx: int, b64_img: str):
        payload = {
            "model": "lightonocr",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": b64_img}
                }]
            }],
            "max_tokens": 4096,
            "temperature": 0.2,
            "top_p": 0.9,
        }

        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return {"status": "success", "data": text}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    with ThreadPoolExecutor(max_workers=min(len(image_b64_list), CONCURRENT_LIMIT)) as exec:
        futures = {exec.submit(process_page, i, img): i for i, img in enumerate(image_b64_list)}
        for f in as_completed(futures):
            idx = futures[f]
            results[f"page_{idx}"] = f.result()

    return results


# -------------------------------------------------
# Local Entrypoint for Testing
# -------------------------------------------------
@app.local_entrypoint()
def main(pdf_url: str = "https://arxiv.org/pdf/2412.13663.pdf"):
    """
    Local test: Uses default PDF. Override with --pdf-url <url>
    Run:
        modal run vllm_lightonocr.py
        modal run vllm_lightonocr.py --pdf-url https://example.com/doc.pdf
    """
    print(f"Testing OCR on PDF: {pdf_url}")

    # Download PDF
    pdf_data = requests.get(pdf_url).content
    pdf = pdfium.PdfDocument(pdf_data)
    n_pages = len(pdf)
    print(f"PDF has {n_pages} pages. Testing first 3...")

    image_b64_list = []
    for i in range(min(n_pages, 3)):
        page = pdf[i]
        pil = page.render(scale=2.77).to_pil()
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        image_b64_list.append(f"data:image/png;base64,{b64}")

    print("Sending to Modal OCR server...")
    results = ocr_batch_pages.remote(image_b64_list)

    print("\n" + "="*60)
    print("OCR RESULTS (first 500 chars per page):")
    print("="*60)
    for k, v in results.items():
        if v["status"] == "success":
            preview = v["data"].replace("\n", " ").strip()[:500]
            print(f"\n{k}:\n{preview}...\n")
        else:
            print(f"{k}: ERROR → {v['error']}")