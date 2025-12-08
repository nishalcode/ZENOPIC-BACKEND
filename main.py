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

# ====== FastAPI App ======
app = FastAPI(title="ZenoPic Backend")

# ====== CORS ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow any frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Root Route ======
@app.get("/")
async def root():
    return {"message": "ZenoPic Backend is live 🚀 Use /generate endpoint"}

# ====== Env Vars ======
HORDE_KEY = os.getenv("HORDE_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_API_KEY")

HORDE_URL = "https://stablehorde.net/api/v2/generate/text2img"
OPENROUTER_URL = "https://openrouter.ai/api/v1/images/generations"
HUGGINGFACE_URL = "https://api-inference.huggingface.co/models/gsdf/Counterfeit-V2.5"

# ====== Rate Limit ======
RATE_LIMIT = 5  # requests per user per minute
user_times = {}

# ====== Queue System ======
task_queue = Queue()
RESULTS = {}

# ====== Pydantic Model ======
class GenRequest(BaseModel):
    prompt: str
    model: str = "horde"  # "horde", "openrouter", "huggingface"
    steps: int = 25
    width: int = 512
    height: int = 512
    cfg_scale: float = 7.5
    sampler: str = "k_euler_ancestral"
    nsfw: bool = False
    return_png: bool = False  # return PNG bytes if True

# ====== Rate limit check ======
def rate_limited(ip: str):
    now = time.time()
    times = user_times.get(ip, [])
    times = [t for t in times if now - t < 60]
    if len(times) >= RATE_LIMIT:
        return True
    times.append(now)
    user_times[ip] = times
    return False

# ====== Image Generation Functions ======
async def generate_horde(req: GenRequest):
    payload = {
        "prompt": req.prompt,
        "params": {
            "steps": req.steps,
            "width": req.width,
            "height": req.height,
            "cfg_scale": req.cfg_scale,
            "sampler_name": req.sampler,
        },
        "nsfw": req.nsfw,
    }
    headers = {"apikey": HORDE_KEY}
    async with httpx.AsyncClient(timeout=250) as client:
        r = await client.post(HORDE_URL, json=payload, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=r.text)
    data = r.json()
    return data["generations"][0]["img"]

async def generate_openrouter(req: GenRequest):
    payload = {
        "model": "SDXL",
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
    if not HUGGINGFACE_KEY:
        raise HTTPException(status_code=403, detail="HuggingFace key required")
    headers = {"Authorization": f"Bearer {HUGGINGFACE_KEY}"}
    payload = {
        "inputs": req.prompt,
        "parameters": {"width": req.width, "height": req.height, "guidance_scale": req.cfg_scale}
    }
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(HUGGINGFACE_URL, json=payload, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=r.text)
    data = r.json()
    return data[0]["generated_image"]

# ====== Main generate function with fallback ======
async def generate(req: GenRequest):
    try:
        if req.model.lower() == "horde":
            return await generate_horde(req)
        elif req.model.lower() == "openrouter":
            return await generate_openrouter(req)
        elif req.model.lower() == "huggingface":
            return await generate_huggingface(req)
        else:
            raise HTTPException(status_code=400, detail="Unknown model")
    except Exception:
        # fallback: try Horde if not Horde
        if req.model.lower() != "horde":
            return await generate_horde(req)
        raise

# ====== Queue worker ======
def worker():
    while True:
        task_id, req = task_queue.get()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            img_b64 = loop.run_until_complete(generate(req))
            RESULTS[task_id] = img_b64
        except Exception as e:
            RESULTS[task_id] = f"ERROR: {e}"
        finally:
            task_queue.task_done()

Thread(target=worker, daemon=True).start()

# ====== API Endpoint ======
@app.post("/generate")
async def api_generate(req: GenRequest, request: Request):
    ip = request.client.host
    if rate_limited(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    task_id = str(time.time_ns())
    task_queue.put((task_id, req))
    
    # Wait for task completion
    while task_id not in RESULTS:
        await asyncio.sleep(0.5)
    
    img_b64 = RESULTS.pop(task_id)
    
    if req.return_png:
        img_bytes = base64.b64decode(img_b64)
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
    
    return {"image": img_b64}
