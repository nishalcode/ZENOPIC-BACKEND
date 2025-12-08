import os
import base64
import io
import asyncio
import time
from queue import Queue
from threading import Thread
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===== FastAPI =====
app = FastAPI(title="ZenoPic Ultimate Backend")

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Root =====
@app.get("/")
async def root():
    return {"message": "ZenoPic Ultimate Backend is live 🚀 Use /generate"}

# ===== Env Vars =====
HORDE_KEY = os.getenv("HORDE_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_BASE = "https://api-inference.huggingface.co/models/"

HORDE_URL = "https://stablehorde.net/api/v2/generate/text2img"
OPENROUTER_URL = "https://openrouter.ai/api/v1/images/generations"

# ===== Rate Limit =====
RATE_LIMIT = 5  # requests per minute per IP
user_times = {}

# ===== Queue =====
task_queue = Queue()
RESULTS = {}

# ===== Pydantic model =====
class GenRequest(BaseModel):
    prompt: str
    service: str = "horde"       # horde / openrouter / huggingface
    model: str = ""              # actual model inside service
    steps: int = 25
    width: int = 512
    height: int = 512
    nsfw: bool = False
    return_png: bool = False

# ===== Rate limit =====
def rate_limited(ip: str):
    now = time.time()
    times = user_times.get(ip, [])
    times = [t for t in times if now - t < 60]
    if len(times) >= RATE_LIMIT:
        return True
    times.append(now)
    user_times[ip] = times
    return False

# ===== Generate functions =====
async def generate_horde(req: GenRequest):
    payload = {
        "prompt": req.prompt,
        "params": {"steps": req.steps, "width": req.width, "height": req.height, "cfg_scale":7.5, "sampler_name":"k_euler_ancestral"},
        "nsfw": req.nsfw
    }
    headers = {"apikey": HORDE_KEY}
    async with httpx.AsyncClient(timeout=250) as client:
        r = await client.post(HORDE_URL, json=payload, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=r.text)
    data = r.json()
    return data["generations"][0]["img"]

async def generate_openrouter(req: GenRequest):
    if not req.model:
        raise HTTPException(status_code=400, detail="OpenRouter requires a model name")
    payload = {
        "model": req.model,
        "prompt": req.prompt,
        "height": req.height,
        "width": req.width,
        "steps": req.steps
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_KEY}"}
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(OPENROUTER_URL, json=payload, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=r.text)
    data = r.json()
    return data["images"][0]

async def generate_huggingface(req: GenRequest):
    if not req.model:
        raise HTTPException(status_code=400, detail="HuggingFace requires a model name")
    headers = {"Authorization": f"Bearer {HUGGINGFACE_KEY}"}
    url = f"{HUGGINGFACE_BASE}{req.model}"
    payload = {"inputs": req.prompt, "parameters":{"width":req.width,"height":req.height,"guidance_scale":7.5}}
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(url, json=payload, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=r.text)
    data = r.json()
    return data[0]["generated_image"]

# ===== Worker =====
def worker():
    while True:
        task_id, req = task_queue.get()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            service = req.service.lower()
            if service == "horde":
                img_b64 = loop.run_until_complete(generate_horde(req))
            elif service == "openrouter":
                img_b64 = loop.run_until_complete(generate_openrouter(req))
            elif service == "huggingface":
                img_b64 = loop.run_until_complete(generate_huggingface(req))
            else:
                img_b64 = f"ERROR: Unknown service {req.service}"
            RESULTS[task_id] = img_b64
        except Exception as e:
            RESULTS[task_id] = f"ERROR: {e}"
        finally:
            task_queue.task_done()

Thread(target=worker, daemon=True).start()

# ===== API endpoint =====
@app.post("/generate")
async def api_generate(req: GenRequest, request: Request):
    ip = request.client.host
    if rate_limited(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    task_id = str(time.time_ns())
    task_queue.put((task_id, req))

    while task_id not in RESULTS:
        await asyncio.sleep(0.5)

    img_b64 = RESULTS.pop(task_id)

    if req.return_png:
        img_bytes = base64.b64decode(img_b64)
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")

    return {"image": img_b64}
