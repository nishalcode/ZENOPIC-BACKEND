import os
import io
import time
import uuid
import base64
import atexit
import logging
from datetime import datetime, timezone, timedelta
from threading import Lock, Thread
from queue import Queue, Empty, Full
import asyncio

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

# ==================== Logging ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== App ====================
app = FastAPI(title="ZenoPic Backend", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== API Keys ====================
HORDE_KEY = os.getenv("HORDE_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_API_KEY")

SERVICES = {
    "horde": ["stable_diffusion_xl", "dreamshaper_8", "midjourney_diffusion"],
    "openrouter": ["stabilityai/stable-diffusion-xl", "black-forest-labs/flux-schnell"],
    "huggingface": ["stabilityai/sdxl-turbo", "runwayml/stable-diffusion-v1-5"]
}

# ==================== Limits ====================
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "25"))
RESULT_TTL = int(os.getenv("RESULT_TTL", "600"))  # seconds
TASK_TIMEOUT = int(os.getenv("TASK_TIMEOUT", "300"))  # seconds
MAX_QUEUE_SIZE = 100
USER_CLEANUP_DAYS = 7

# ==================== State ====================
user_usage = {}
usage_lock = Lock()

task_queue = Queue(maxsize=MAX_QUEUE_SIZE)
RESULTS = {}
RESULT_TIMESTAMPS = {}
TASK_META = {}
results_lock = Lock()

shutdown_flag = False

# ==================== Startup ====================
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 ZenoPic Backend starting...")
    logger.info(f"✅ API Keys: Horde={bool(HORDE_KEY)}, OpenRouter={bool(OPENROUTER_KEY)}, HuggingFace={bool(HUGGINGFACE_KEY)}")

# ==================== Helpers ====================
def check_and_increment_usage(user_id: str) -> bool:
    now = datetime.now(timezone.utc)
    with usage_lock:
        if user_id in user_usage:
            count, reset_time = user_usage[user_id]
            if now >= reset_time:
                user_usage[user_id] = (1, now + timedelta(days=1))
                return True
            if count >= DAILY_LIMIT:
                return False
            user_usage[user_id] = (count + 1, reset_time)
            return True
        else:
            user_usage[user_id] = (1, now + timedelta(days=1))
            return True

def get_usage(user_id: str) -> tuple:
    with usage_lock:
        if user_id in user_usage:
            return user_usage[user_id]
        return (0, datetime.now(timezone.utc) + timedelta(days=1))

def cleanup_old_results():
    now = time.time()
    with results_lock:
        to_remove = [
            task_id for task_id, timestamp in list(RESULT_TIMESTAMPS.items())
            if now - timestamp > RESULT_TTL
        ]
        for task_id in to_remove:
            RESULTS.pop(task_id, None)
            RESULT_TIMESTAMPS.pop(task_id, None)
            TASK_META.pop(task_id, None)
        if to_remove:
            logger.info(f"🧹 Cleaned up {len(to_remove)} old results")

def cleanup_old_users():
    now = datetime.now(timezone.utc)
    with usage_lock:
        to_delete = [
            user_id for user_id, (_, reset_time) in list(user_usage.items())
            if now - reset_time > timedelta(days=USER_CLEANUP_DAYS)
        ]
        for user_id in to_delete:
            del user_usage[user_id]
        if to_delete:
            logger.info(f"🧹 Cleaned up {len(to_delete)} old user entries")

def validate_base64(b64_string: str) -> str:
    if not b64_string:
        raise ValueError("Empty base64 string")
    if b64_string.startswith("data:image"):
        b64_string = b64_string.split(",", 1)[1] if "," in b64_string else b64_string
    # Simplified padding
    b64_string += "=" * (-len(b64_string) % 4)
    try:
        base64.b64decode(b64_string, validate=True)
        return b64_string
    except Exception as e:
        raise ValueError(f"Invalid base64 data: {str(e)}")

def validate_model(service: str, model: str) -> bool:
    if not model or not model.strip():
        return False
    return service in SERVICES and model in SERVICES[service]

# ==================== Models ====================
class GenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    service: str = Field(..., pattern="^(horde|openrouter|huggingface)$")
    model: str = Field(..., min_length=1)
    steps: int = Field(default=25, ge=1, le=100)
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    return_png: bool = Field(default=False)

    @field_validator("service")
    @classmethod
    def normalize_service(cls, v):
        return v.lower()

# ==================== AI Generators ====================
async def generate_horde(req: GenRequest) -> str:
    if not HORDE_KEY:
        return "ERROR: HORDE_API_KEY not configured"
    url = "https://aihorde.net/api/v2/generate/async"
    payload = {"prompt": req.prompt, "params": {"width": req.width, "height": req.height, "steps": req.steps}, "models": [req.model]}
    headers = {"apikey": HORDE_KEY}
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        try:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            job_id = data.get("id")
            if not job_id:
                return "ERROR: No job ID from Horde"
            poll_url = f"https://aihorde.net/api/v2/generate/status/{job_id}"
            start_poll = time.time()
            for attempt in range(60):
                elapsed = time.time() - start_poll
                await asyncio.sleep(0.5 if elapsed < 5 else 1 if elapsed < 30 else 2)
                status = await client.get(poll_url, headers=headers)
                status.raise_for_status()
                sdata = status.json()
                if sdata.get("done"):
                    gens = sdata.get("generations", [])
                    if gens:
                        gen = gens[0]
                        for key in ["img", "image", "base64", "b64_json"]:
                            if key in gen and gen[key]:
                                return str(gen[key])
                    return "ERROR: No image in Horde response"
            return "ERROR: Horde generation timeout"
        except Exception as e:
            return f"ERROR: {str(e)}"

async def generate_openrouter(req: GenRequest) -> str:
    if not OPENROUTER_KEY:
        return "ERROR: OPENROUTER_API_KEY not configured"
    url = "https://openrouter.ai/api/v1/images/generations"
    payload = {"model": req.model, "prompt": req.prompt, "steps": req.steps, "width": req.width, "height": req.height, "num_images":1}
    headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        try:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            if "data" in data and data["data"]:
                for key in ["b64_json", "base64", "image"]:
                    if key in data["data"][0] and data["data"][0][key]:
                        return str(data["data"][0][key])
            return "ERROR: Could not extract image from OpenRouter response"
        except Exception as e:
            return f"ERROR: {str(e)}"

async def generate_huggingface(req: GenRequest) -> str:
    if not HUGGINGFACE_KEY:
        return "ERROR: HUGGINGFACE_API_KEY not configured"
    url = f"https://api-inference.huggingface.co/models/{req.model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_KEY}"}
    payload = {"inputs": req.prompt, "parameters": {"width": req.width, "height": req.height, "num_inference_steps": req.steps}, "options": {"wait_for_model": True}}
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        try:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            if ct.startswith("image/"):
                return base64.b64encode(r.content).decode("utf-8")
            try:
                data = r.json()
                if isinstance(data, list) and data:
                    item = data[0]
                    if isinstance(item, dict):
                        for key in ["generated_image", "image", "b64_json"]:
                            if key in item and item[key]:
                                return str(item[key])
            except:
                if r.content:
                    return base64.b64encode(r.content).decode("utf-8")
            return "ERROR: Could not extract image from HuggingFace response"
        except Exception as e:
            return f"ERROR: {str(e)}"

# ==================== Worker ====================
def _worker_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    logger.info("🚀 Worker thread started")
    cleanup_counter = 0

    async def process_task(task_id: str, req: GenRequest) -> str:
        try:
            if req.service == "horde":
                return await generate_horde(req)
            elif req.service == "openrouter":
                return await generate_openrouter(req)
            elif req.service == "huggingface":
                return await generate_huggingface(req)
            else:
                return f"ERROR: Unknown service {req.service}"
        except Exception as e:
            return f"ERROR: {str(e)}"

    while not shutdown_flag:
        try:
            task_id, req = task_queue.get(timeout=1)
        except Empty:
            cleanup_counter += 1
            if cleanup_counter >= 10:
                cleanup_old_results()
                cleanup_old_users()
                cleanup_counter = 0
            continue

        try:
            img_b64 = loop.run_until_complete(process_task(task_id, req))
            task_queue.task_done()
            with results_lock:
                RESULTS[task_id] = img_b64
                RESULT_TIMESTAMPS[task_id] = time.time()
                if task_id in TASK_META:
                    TASK_META[task_id]["completed"] = True
                else:
                    TASK_META[task_id] = {"completed": True, "timestamp": time.time()}
            logger.info(f"✅ Task {task_id[:8]} completed")
        except Exception as e:
            logger.error(f"❌ Worker error {task_id[:8]}: {e}")
            task_queue.task_done()
            with results_lock:
                RESULTS[task_id] = f"ERROR: Worker error - {str(e)}"
                RESULT_TIMESTAMPS[task_id] = time.time()
                if task_id in TASK_META:
                    TASK_META[task_id]["completed"] = True
                else:
                    TASK_META[task_id] = {"completed": True, "timestamp": time.time()}

    logger.info("🛑 Worker shutting down")
    loop.close()

worker_thread = Thread(target=_worker_loop, daemon=True, name="ImageGenWorker")
worker_thread.start()

# ==================== Endpoints ====================
@app.get("/")
async def root():
    return {"status":"online","message":"ZenoPic Backend","version":"1.0.1","services":list(SERVICES.keys())}

@app.get("/models/{service}")
async def get_models(service: str):
    service = service.lower()
    if service not in SERVICES:
        raise HTTPException(status_code=404, detail="Service not found")
    return {"service": service, "models": SERVICES[service]}

@app.get("/usage/{user_id}")
async def get_usage_endpoint(user_id: str):
    count, reset_time = get_usage(user_id)
    return {"user_id": user_id, "count": count, "limit": DAILY_LIMIT, "reset_time": reset_time.isoformat(), "remaining": max(0, DAILY_LIMIT-count)}

@app.get("/health")
async def health():
    with results_lock:
        queue_size = task_queue.qsize()
    return {
        "status":"healthy",
        "queue_size":queue_size,
        "queue_max":MAX_QUEUE_SIZE,
        "queue_available":MAX_QUEUE_SIZE-queue_size,
        "pending_results":len(RESULTS),
        "active_tasks":len([m for m in TASK_META.values() if not m.get("completed", False)]),
        "total_users":len(user_usage),
        "worker_alive":worker_thread.is_alive()
    }

@app.post("/generate")
async def generate(req: GenRequest, request: Request):
    user_id = request.client.host if request.client else "unknown"
    if not validate_model(req.service, req.model):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{req.model}' for service '{req.service}'. Available models: {SERVICES.get(req.service, [])}"
        )
    if not check_and_increment_usage(user_id):
        count, reset_time = get_usage(user_id)
        raise HTTPException(status_code=429, detail=f"Daily limit reached ({DAILY_LIMIT}) resets at {reset_time.isoformat()}")
    if task_queue.qsize() >= MAX_QUEUE_SIZE-5:
        raise HTTPException(status_code=503, detail="Server busy, try again later")

    task_id = str(uuid.uuid4())
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"📥 Request {request_id} from {user_id}: {req.service}/{req.model}")

    with results_lock:
        TASK_META[task_id] = {"timestamp": time.time(), "request_id": request_id, "user_id": user_id, "service": req.service, "model": req.model, "completed": False}

    try:
        task_queue.put((task_id, req), timeout=2)
    except Full:
        with results_lock:
            TASK_META.pop(task_id, None)
        raise HTTPException(status_code=503, detail="Queue full")

    start_time = time.time()
    img_b64 = None
    while True:
        with results_lock:
            if task_id in RESULTS:
                img_b64 = RESULTS.pop(task_id)
                RESULT_TIMESTAMPS.pop(task_id, None)
                TASK_META.pop(task_id, None)
                break
        if time.time() - start_time > TASK_TIMEOUT:
            with results_lock:
                RESULTS.pop(task_id, None)
                RESULT_TIMESTAMPS.pop(task_id, None)
                TASK_META.pop(task_id, None)
            raise HTTPException(status_code=504, detail="Generation timed out")
        await asyncio.sleep(0.2 if time.time()-start_time<5 else 0.5 if time.time()-start_time<30 else 1)

    if not img_b64 or (isinstance(img_b64,str) and img_b64.startswith("ERROR")):
        error_msg = img_b64.replace("ERROR: ","") if isinstance(img_b64,str) else "Generation failed"
        raise HTTPException(status_code=500, detail=error_msg)

    try:
        img_b64 = validate_base64(img_b64)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if req.return_png:
        image_bytes = base64.b64decode(img_b64)
        return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png",
                                 headers={"X-Request-ID":request_id,"Cache-Control":"public, max-age=31536000","Content-Disposition":f"inline; filename=zenopic_{request_id}.png"})
    return JSONResponse(content={"request_id":request_id,"service":req.service,"model":req.model,"image":img_b64}, headers={"X-Request-ID":request_id,"Cache-Control":"no-cache"})

# ==================== Shutdown ====================
def cleanup():
    global shutdown_flag
    shutdown_flag = True
    logger.info("🛑 Shutting down...")
    if worker_thread.is_alive():
        worker_thread.join(timeout=10)
        if worker_thread.is_alive():
            logger.warning("Worker thread did not terminate gracefully")
    with results_lock:
        RESULTS.clear()
        RESULT_TIMESTAMPS.clear()
        TASK_META.clear()
    logger.info("✅ Shutdown complete")

atexit.register(cleanup)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
