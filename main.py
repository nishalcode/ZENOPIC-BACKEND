import os, io, time, uuid, base64, json, logging, asyncio, pickle, hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from collections import OrderedDict

import httpx
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

# ---------------- Config ----------------
POLLINATIONS_BASE = os.getenv("POLLINATIONS_BASE","https://api.pollinations.ai/image")
LOCAL_ENDPOINT = os.getenv("LOCAL_SD_ENDPOINT")
PORT = int(os.getenv("PORT","8000"))
CREDITS_ENABLED = os.getenv("CREDITS_ENABLED","true").lower() in ("1","true","yes")
FREE_CREDITS = int(os.getenv("FREE_CREDITS","20"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET")  # MUST set in production
CACHE_TTL = int(os.getenv("CACHE_TTL","3600"))
CACHE_CLEAN_INTERVAL = int(os.getenv("CACHE_CLEAN_INTERVAL","300"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY","3"))
QUEUE_MAXSIZE = int(os.getenv("QUEUE_MAXSIZE","200"))
WATERMARK_ENABLED = os.getenv("WATERMARK_ENABLED","true").lower() in ("1","true","yes")
WATERMARK_TEXT = os.getenv("WATERMARK_TEXT","NGAI")
WATERMARK_OPACITY = float(os.getenv("WATERMARK_OPACITY","0.85"))
WATERMARK_PADDING = int(os.getenv("WATERMARK_PADDING","10"))
MAX_REQUEST_SIZE = 10*1024*1024

SERVICES={"pollinations":["flux-dev","flux-realism","flux-anime","flux-pro","turbo","default"],"local":["self-hosted"]}

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")
logger=logging.getLogger("zenopic-ultra")

# ---------------- App ----------------
app=FastAPI(title="ZenoPic Backend — Ultra",version="3.1")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

# ---------------- State ----------------
TASK_QUEUE: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
TASKS: Dict[str, Dict[str, Any]] = {}
RESULTS: Dict[str, Dict[str, Any]] = {}
LOG_ENTRIES: list = []
RATE_STORE: Dict[str, Dict[str, Any]] = {}
shutdown_flag=False

class LRUCache:
    def __init__(self,capacity=1000): self.cache=OrderedDict(); self.capacity=capacity
    def get(self,key): 
        if key not in self.cache: return None
        self.cache.move_to_end(key)
        return self.cache[key]
    def set(self,key,value):
        if key in self.cache: self.cache.move_to_end(key)
        self.cache[key]=value
        if len(self.cache)>self.capacity: self.cache.popitem(last=False)

CACHE=LRUCache(capacity=1000)
CREDITS: Dict[str,int]={}

# ---------------- Models ----------------
class GenRequest(BaseModel):
    prompt:str=Field(...,min_length=1,max_length=2000)
    service:str=Field(...,pattern="^(pollinations|local)$")
    model:str=Field(...,min_length=1)
    steps:int=Field(default=28,ge=1,le=200)
    width:int=Field(default=512,ge=64,le=2048)
    height:int=Field(default=512,ge=64,le=2048)
    cfg_scale:float=Field(default=7.0,ge=1.0,le=30.0)
    negative_prompt:str=Field(default="")
    seed:int=Field(default=-1,ge=-1)
    return_png:bool=Field(default=False)
    @field_validator("service")
    @classmethod
    def norm_service(cls,v): return v.lower()

# ---------------- Helpers ----------------
def make_cache_key(req:GenRequest)->str:
    keydata={"prompt":req.prompt.strip(),"service":req.service,"model":req.model,
             "steps":req.steps,"w":req.width,"h":req.height,"cfg":req.cfg_scale,
             "neg":req.negative_prompt or "","seed":req.seed}
    raw=json.dumps(keydata,sort_keys=True,ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]

def cache_get(key:str)->Optional[str]:
    item=CACHE.get(key)
    if not item: return None
    if time.time()-item["ts"]>CACHE_TTL: return None  # don't pop here
    return item["b64"]

def cache_set(key:str,b64:str): CACHE.set(key,{"ts":time.time(),"b64":b64})

def log_request(entry:Dict[str,Any]):
    entry["ts"]=datetime.now(timezone.utc).isoformat()
    LOG_ENTRIES.append(entry)
    if len(LOG_ENTRIES)>200: LOG_ENTRIES.pop(0)

def ensure_credits(user_id:str):
    if user_id not in CREDITS: CREDITS[user_id]=FREE_CREDITS

def consume_credit(user_id:str)->bool:
    if not CREDITS_ENABLED: return True
    ensure_credits(user_id)
    if CREDITS[user_id]<=0: return False
    CREDITS[user_id]-=1
    return True

def add_watermark_to_image_bytes(img_bytes:bytes)->bytes:
    try:
        img=Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        draw=ImageDraw.Draw(img)
        try: font=ImageFont.truetype("arial.ttf",max(12,img.width//40))
        except: font=ImageFont.load_default()
        text=WATERMARK_TEXT
        bbox=draw.textbbox((0,0),text,font=font)
        text_w=bbox[2]-bbox[0]; text_h=bbox[3]-bbox[1]
        x=img.width-text_w-WATERMARK_PADDING
        y=img.height-text_h-WATERMARK_PADDING
        draw.text((x+1,y+1),text,fill=(0,0,0,int(255*WATERMARK_OPACITY)),font=font)
        draw.text((x,y),text,fill=(255,255,255,int(255*WATERMARK_OPACITY)),font=font)
        out=io.BytesIO(); img.convert("RGB").save(out,format="PNG"); return out.getvalue()
    except Exception as e: logger.warning(f"Watermark failed: {e}"); return img_bytes

# ---------------- Rate Limiting ----------------
RATE_LIMIT=10; RATE_WINDOW=60
def rate_limit(ip:str)->bool:
    now=time.time()
    state=RATE_STORE.get(ip,{"count":0,"start":now})
    if now-state["start"]>RATE_WINDOW: state={"count":0,"start":now}
    state["count"]+=1; RATE_STORE[ip]=state
    return state["count"]<=RATE_LIMIT

async def cleanup_rate_store():
    while not shutdown_flag:
        now=time.time()
        to_remove=[ip for ip,state in list(RATE_STORE.items()) if now-state.get("start",0)>86400]
        for ip in to_remove: RATE_STORE.pop(ip,None)
        await asyncio.sleep(3600)

# ---------------- Generators ----------------
async def call_pollinations(req:GenRequest)->str:
    if req.model not in SERVICES.get("pollinations",[]): return f"ERROR: Invalid model '{req.model}'"
    params={"prompt":req.prompt,"model":req.model,"width":req.width,"height":req.height,"steps":req.steps,"nologo":"true"}
    if req.seed>=0: params["seed"]=str(req.seed)
    if req.negative_prompt: params["negative_prompt"]=req.negative_prompt
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        try:
            r=await client.get(POLLINATIONS_BASE,params=params); r.raise_for_status(); content=r.content
            if not content: return "ERROR: Empty response"
            if WATERMARK_ENABLED: content=add_watermark_to_image_bytes(content)
            return base64.b64encode(content).decode("utf-8")
        except Exception as e: return f"ERROR:{e}"

async def call_local(req:GenRequest)->str:
    if not LOCAL_ENDPOINT: return "ERROR: LOCAL_SD_ENDPOINT not set"
    if req.model not in SERVICES.get("local",[]): return f"ERROR: Invalid model '{req.model}'"
    payload={"prompt":req.prompt,"negative_prompt":req.negative_prompt,"steps":req.steps,"width":req.width,"height":req.height,"cfg_scale":req.cfg_scale,"sampler_name":"Euler a","n_iter":1,"batch_size":1}
    if req.seed>=0: payload["seed"]=req.seed
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        try:
            r=await client.post(LOCAL_ENDPOINT,json=payload); r.raise_for_status()
            data=r.json(); imgs=data.get("images") or data.get("output") or []
            if imgs: content=base64.b64decode(str(imgs[0]))
            else: return "ERROR: No image returned"
            if WATERMARK_ENABLED: content=add_watermark_to_image_bytes(content)
            return base64.b64encode(content).decode("utf-8")
        except Exception as e: return f"ERROR:{e}"

# ---------------- Worker ----------------
async def worker_loop():
    while not shutdown_flag:
        try: task_id, req_dict, user_id = await TASK_QUEUE.get()
        except asyncio.CancelledError: break
        try: req=GenRequest(**req_dict)
        except Exception as e:
            RESULTS[task_id]={"status":"failed","error":f"Invalid request: {e}"}
            TASKS[task_id]["status"]="failed"
            TASK_QUEUE.task_done()
            continue
        TASKS[task_id]["status"]="processing"
        cache_key=make_cache_key(req)
        cached=cache_get(cache_key)
        if cached:
            RESULTS[task_id]={"b64":cached,"status":"complete","cached":True}
            TASKS[task_id]["status"]="complete"
            TASK_QUEUE.task_done()
            log_request({"task_id":task_id,"user":user_id,"status":"cached"})
            continue
        ok=consume_credit(user_id)
        if not ok:
            RESULTS[task_id]={"status":"failed","error":"No credits"}
            TASKS[task_id]["status"]="failed"
            TASK_QUEUE.task_done()
            log_request({"task_id":task_id,"user":user_id,"status":"failed_no_credits"})
            continue
        if req.service=="pollinations": result=await call_pollinations(req)
        else: result=await call_local(req)
        if result.startswith("ERROR"):
            RESULTS[task_id]={"status":"failed","error":result}
            TASKS[task_id]["status"]="failed"
        else:
            RESULTS[task_id]={"b64":result,"status":"complete"}
            TASKS[task_id]["status"]="complete"
            cache_set(cache_key,result)
        TASK_QUEUE.task_done()

# ---------------- Background ----------------
async def cache_cleaner():
    while not shutdown_flag:
        now=time.time()
        to_remove=[k for k,v in CACHE.cache.items() if now-v["ts"]>CACHE_TTL]
        for k in to_remove: CACHE.cache.pop(k,None)
        await asyncio.sleep(CACHE_CLEAN_INTERVAL)

async def cleanup_old_tasks():
    while not shutdown_flag:
        now=time.time(); cutoff=now-86400
        for task_id,meta in list(TASKS.items()):
            created=meta.get("created_at")
            try: created_ts=datetime.fromisoformat(created).timestamp() if created else 0
            except: created_ts=0
            if created_ts<cutoff: TASKS.pop(task_id,None); RESULTS.pop(task_id,None)
        await asyncio.sleep(3600)

# ---------------- Persistence ----------------
STATE_FILE="zenopic_state.pkl"
def load_state():
    try:
        with open(STATE_FILE,"rb") as f:
            data=pickle.load(f)
            CREDITS.update(data.get("CREDITS",{}))
            for k,v in data.get("CACHE",{}).items():
                if time.time()-v["ts"]<CACHE_TTL: CACHE.set(k,v)
    except FileNotFoundError: pass

def save_state():
    with open(STATE_FILE,"wb") as f:
        pickle.dump({"CREDITS":CREDITS,"CACHE":{k:v for k,v in CACHE.cache.items() if time.time()-v["ts"]<CACHE_TTL}},f)

# ---------------- Startup ----------------
@app.on_event("startup")
async def startup_background_jobs():
    load_state()
    for demo in ("demo_user",): ensure_credits(demo)
    asyncio.create_task(cache_cleaner())
    asyncio.create_task(cleanup_old_tasks())
    asyncio.create_task(cleanup_rate_store())
    for _ in range(MAX_CONCURRENCY): asyncio.create_task(worker_loop())

@app.on_event("shutdown")
async def shutdown():
    global shutdown_flag; shutdown_flag=True
    save_state()

# ---------------- Endpoints ----------------
@app.get("/")
async def root(): return {"status":"online","services":list(SERVICES.keys())}

@app.get("/services")
async def services(): return {"services":list(SERVICES.keys())}

@app.get("/models/{service}")
async def models(service:str):
    s=service.lower()
    if s not in SERVICES: raise HTTPException(status_code=404,detail="Service not found")
    return {"service":s,"models":SERVICES[s]}

@app.get("/credits/{user_id}")
async def get_credits(user_id:str):
    ensure_credits(user_id)
    return {"user_id":user_id,"credits":CREDITS.get(user_id,0),"enabled":CREDITS_ENABLED}

@app.post("/credits/topup")
async def topup_credits(user_id:str,amount:int=10,admin_secret:Optional[str]=None):
    if not ADMIN_SECRET: raise HTTPException(status_code=501,detail="Admin endpoint not configured")
    if admin_secret!=ADMIN_SECRET: raise HTTPException(status_code=401,detail="Invalid admin secret")
    ensure_credits(user_id)
    CREDITS[user_id]+=amount
    return {"user_id":user_id,"credits":CREDITS[user_id]}

@app.post("/generate")
async def generate(req:GenRequest,request:Request):
    client_host=request.client.host if request.client else "unknown"
    user_id=f"user:{client_host}"
    if not rate_limit(client_host): return JSONResponse(status_code=429,content={"detail":"Rate limit exceeded"})
    ensure_credits(user_id)
    if CREDITS_ENABLED and CREDITS.get(user_id,0)<=0: raise HTTPException(status_code=402,detail="No credits remaining")
    cache_key=make_cache_key(req); cached=cache_get(cache_key)
    if cached: return {"task_id":"cached","status":"complete","image":cached,"cached":True}
    task_id=str(uuid.uuid4())
    TASKS[task_id]={"status":"queued","created_at":datetime.now(timezone.utc).isoformat(),"user":user_id,"prompt":req.prompt,"service":req.service,"model":req.model}
    try: await asyncio.wait_for(TASK_QUEUE.put((task_id,req.dict(),user_id)),timeout=2)
    except asyncio.TimeoutError: TASKS.pop(task_id,None); raise HTTPException(status_code=503,detail="Queue full")
    log_request({"task_id":task_id,"user":user_id,"status":"queued"})
    return {"task_id":task_id,"status":"queued"}

@app.get("/status/{task_id}")
async def status(task_id:str):
    if task_id not in TASKS: raise HTTPException(status_code=404,detail="Task not found")
    meta=TASKS[task_id].copy(); meta["result"]=RESULTS.get(task_id)
    return meta

@app.get("/result/{task_id}")
async def result(task_id:str,as_png:Optional[bool]=False):
    entry=RESULTS.get(task_id)
    if not entry: raise HTTPException(status_code=404,detail="Result not found")
    if entry.get("status")!="complete": raise HTTPException(status_code=400,detail=f"Task status: {entry.get('status')}")
    b64=entry.get("b64")
    if not b64: raise HTTPException(status_code=500,detail="Missing image data")
    img_bytes=base64.b64decode(b64)
    return StreamingResponse(io.BytesIO(img_bytes),media_type="image/png") if as_png else {"task_id":task_id,"image":b64,"cached":entry.get("cached",False)}

# ---------------- Run ----------------
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=PORT,log_level="info")
