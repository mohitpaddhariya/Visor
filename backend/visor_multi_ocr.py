# --------------------------------------------------------------
# visor_multi_ocr.py
# --------------------------------------------------------------
import modal
import os
import base64
import io
import json
import requests
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------------------------------------------
# GLOBAL CONFIG
# --------------------------------------------------------------
HF_SECRET_NAME = "huggingface-secret"
GPU = "L40S"
NUMBER_OF_GPUS = 1
GPU_CONFIG = os.environ.get("GPU_CONFIG", f"{GPU}:{NUMBER_OF_GPUS}")
VLLM_PORT = 8000
CONCURRENT_LIMIT = 8                     # max parallel pages per container
FAST_BOOT = False

# ---------- DotsOCR ----------
DOTS_MODEL_NAME = "rednote-hilab/dots.ocr"
DOTS_MODEL_DIR = "DotsOCR"
DOTS_SERVED_NAME = "dotsocr-model"
DOTS_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
3. Text Extraction & Formatting Rules:
   - Picture: omit the text field.
   - Formula: format as LaTeX.
   - Table: format as HTML.
   - All others: format as Markdown.
4. Constraints:
   - Keep the original text (no translation).
   - Sort elements in reading order.
5. Final Output: a **single JSON list** of layout elements."""

# ---------- LightOnOCR ----------
LIGHTON_MODEL_NAME = "lightonai/LightOnOCR-1B-1025"
LIGHTON_MODEL_DIR = "LightOnOCR-1B-1025"
LIGHTON_SERVED_NAME = "lightonocr"

# --------------------------------------------------------------
# IMAGE BUILDERS
# --------------------------------------------------------------
# DotsOCR – vLLM 0.9.1 (needs flash-attn, custom entrypoint patch)
dots_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev", "git")
    .run_commands("pip install uv")
    .run_commands("uv pip install vllm==0.9.1 --system")
    .uv_pip_install(
        "torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0",
        "transformers==4.51.3", "pillow", "pymupdf",
        "huggingface_hub[hf_transfer]", "requests", "numpy", "qwen-vl-utils"
    )
    .run_commands(
        "uv pip install flash-attn==2.8.0.post2 --no-build-isolation --system"
    )
    # ← YOUR ORIGINAL CLONE & EDITABLE INSTALL
    .run_commands(
        "git clone https://github.com/rednote-hilab/dots.ocr.git /tmp/dots_ocr && "
        "cd /tmp/dots_ocr && "
        "uv pip install -e . --system"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/data/.cache",
        # we will set PYTHONPATH **inside** the container later
    })
)

# LightOnOCR – latest nightly vLLM (supports async-scheduling)
lighton_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev", "git", "build-essential")
    .run_commands("pip install uv")
    .run_commands(
        "uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 "
        "--index-url https://download.pytorch.org/whl/cu124 --system"
    )
    .run_commands(
        "uv pip install vllm --system "
        "--extra-index-url https://wheels.vllm.ai/nightly --prerelease=allow"
    )
    .run_commands(
        "uv pip install transformers pillow pymupdf huggingface_hub[hf_transfer] "
        "requests numpy pypdfium2 --system"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/data/.cache"})
)

# --------------------------------------------------------------
# PERSISTENT VOLUME (shared cache)
# --------------------------------------------------------------
volume = modal.Volume.from_name("visor-model-cache", create_if_missing=True)

app = modal.App("visor-multi-ocr", image=dots_image)   # any image works for the app object

# --------------------------------------------------------------
# ---------- DOTS OCR SERVER ----------
# --------------------------------------------------------------
@app.function(
    image=dots_image,
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={"/data": volume},
    scaledown_window=3 * 60,
    max_containers=2,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=CONCURRENT_LIMIT)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def dots_serve():
    import subprocess, os

    print(f"Starting DotsOCR vLLM server with concurrency limit: {CONCURRENT_LIMIT}")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN missing")

    model_path = f"/data/.cache/{DOTS_MODEL_DIR}"
    
    # Download model if not already cached
    if not os.path.exists(model_path):
        print(f"Downloading {DOTS_MODEL_NAME} → {model_path}")
        subprocess.run([
            "python3", "-c", f"""
from huggingface_hub import snapshot_download
snapshot_download(repo_id='{DOTS_MODEL_NAME}', local_dir='{model_path}', token='{hf_token}')
"""
        ], check=True)
        print(f"Model downloaded successfully to {model_path}")
        volume.commit()
    else:
        print(f"Using cached model from {model_path}")

    # Set PYTHONPATH to include the cache directory (where DotsOCR folder is)
    os.environ["PYTHONPATH"] = f"/data/.cache:{os.environ.get('PYTHONPATH', '')}"

    # Register DotsOCR model with vLLM by patching the vllm entrypoint
    print("Registering DotsOCR model with vLLM...")
    patch_cmd = f"""
sed -i '/^from vllm\\.entrypoints\\.cli\\.main import main$/a\\
from {DOTS_MODEL_DIR} import modeling_dots_ocr_vllm' $(which vllm)
"""
    subprocess.run(patch_cmd, shell=True, check=True)

    # Build vLLM server command
    cmd = [
        "vllm", "serve",
        "--uvicorn-log-level=info",
        model_path,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", str(NUMBER_OF_GPUS),
        "--gpu-memory-utilization", "0.8",
        "--chat-template-content-format", "string",
        "--trust-remote-code",
        "--served-model-name", DOTS_SERVED_NAME,
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", "8192",
    ]
    
    if FAST_BOOT:
        cmd += ["--enforce-eager"]
    else:
        cmd += ["--no-enforce-eager"]
    
    print("Starting DotsOCR server with command:")
    print(" ".join(cmd))
    subprocess.Popen(cmd)


# --------------------------------------------------------------
# ---------- LIGHTON OCR SERVER ----------
# --------------------------------------------------------------
@app.function(
    image=lighton_image,
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={"/data": volume},
    scaledown_window=3 * 60,
    max_containers=2,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=CONCURRENT_LIMIT)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def lighton_serve():
    import subprocess, os

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN missing")

    model_path = f"/data/.cache/{LIGHTON_MODEL_DIR}"
    if not os.path.exists(model_path):
        print(f"Downloading {LIGHTON_MODEL_NAME} → {model_path}")
        subprocess.run([
            "python", "-c", f"""
from huggingface_hub import snapshot_download
snapshot_download(repo_id='{LIGHTON_MODEL_NAME}', local_dir='{model_path}', token='{hf_token}')
"""
        ], check=True)
        volume.commit()

    cmd = [
        "vllm", "serve", model_path,
        "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        "--limit-mm-per-prompt", '{"image":1}',
        "--async-scheduling",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.8",
        "--trust-remote-code",
        "--served-model-name", LIGHTON_SERVED_NAME,
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", "8192",
    ]
    if FAST_BOOT:
        cmd += ["--enforce-eager"]
    print("LightOnOCR server:", " ".join(cmd))
    subprocess.Popen(cmd)


# --------------------------------------------------------------
# ---------- MULTI-MODEL BATCH OCR FUNCTION ----------
# --------------------------------------------------------------
@app.function(
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=10 * 60,
)
def ocr_batch_pages(image_b64_list: List[str], model: str = "dotsocr") -> Dict[str, Dict]:
    """
    Multi-Model OCR API – processes a list of base64 PNG images using either DotsOCR or LightOnOCR.

    Parameters
    ----------
    image_b64_list: List[str]
        Each entry: ``data:image/png;base64,<base64-data>``
    model: str
        ``"dotsocr"``  → layout JSON list (DotsOCR)
        ``"lightonocr"`` → markdown string (LightOnOCR)

    Returns
    -------
    Dict[str, Dict]
        {
            "page_0": {"status": "success", "data": <layout list or markdown>},
            ...
        }
    """
    if model not in {"dotsocr", "lightonocr"}:
        raise ValueError("model must be 'dotsocr' or 'lightonocr'")

    # Pick the right server function
    server_fn = dots_serve if model == "dotsocr" else lighton_serve
    served_name = DOTS_SERVED_NAME if model == "dotsocr" else LIGHTON_SERVED_NAME

    try:
        server_url = server_fn.get_web_url()
    except modal.exception.NotFoundError:
        raise RuntimeError(
            f"{model.upper()} server not deployed. Run `modal deploy visor_multi_ocr.py` first."
        )

    if not server_url:
        raise RuntimeError(f"{model.upper()} server is not running.")

    url = f"{server_url}/v1/chat/completions"
    results: Dict[str, Dict] = {}

    print(f"Processing {len(image_b64_list)} pages in parallel using {model.upper()}...")

    def _call_page(idx: int, b64_img: str) -> Dict:
        # ---------- payload construction ----------
        if model == "dotsocr":
            content = [
                {"type": "text", "text": DOTS_PROMPT},
                {"type": "image_url", "image_url": {"url": b64_img}},
            ]
            max_tokens = 8192
            temperature = 0.0
            top_p = 1.0
        else:  # lightonocr
            content = [{"type": "image_url", "image_url": {"url": b64_img}}]
            max_tokens = 4096
            temperature = 0.2
            top_p = 0.9

        payload = {
            "model": served_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]

            if model == "dotsocr":
                data = json.loads(raw)
                if not isinstance(data, list):
                    raise ValueError("DotsOCR did not return a list")
            else:
                data = raw.strip()

            return {"status": "success", "data": data}
        except Exception as e:
            print(f"Error processing page {idx}: {e}")
            return {"status": "error", "error": str(e)}

    # ---------- parallel execution ----------
    max_workers = min(len(image_b64_list), CONCURRENT_LIMIT)
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        future_to_idx = {exe.submit(_call_page, i, img): i for i, img in enumerate(image_b64_list)}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            results[f"page_{i}"] = fut.result()

    print(f"All {len(image_b64_list)} pages processed.")
    return results