import asyncio
import hashlib
import os
import time
from typing import Optional, List, Dict, Tuple

from MySQLModel import MySQLModel
from bin.config_utils import LT_URL, CONCURRENCY, QUEUE_MAX, WAIT_TIMEOUT_S
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


TABLE = "translations_cache"

app = FastAPI(title="Local Translator API", version="1.0")

# kolejka + deduplikacja “in-flight”
queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
inflight: Dict[str, asyncio.Future] = {}
inflight_lock = asyncio.Lock()
sem = asyncio.Semaphore(CONCURRENCY)

db: Optional[MySQLModel] = None
http: Optional[httpx.AsyncClient] = None



class TranslateIn(BaseModel):
    text: str = Field(..., min_length=1)
    source: Optional[str] = None     # None => auto
    target: str = "pl"
    format: str = "text"             # "text" albo "html"


class TranslateOut(BaseModel):
    text: str
    cached: bool
    cache_key: str
    engine: str = "libretranslate"
    ms: int


class BatchIn(BaseModel):
    texts: List[str]
    source: Optional[str] = None
    target: str = "pl"
    format: str = "text"


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_key(text: str, source: Optional[str], target: str, fmt: str) -> str:
    src = (source or "auto").strip().lower()
    dst = target.strip().lower()
    fmt = fmt.strip().lower()
    base = f"{src}|{dst}|{fmt}|{_sha256(text)}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


async def _db_get(cache_key: str) -> Optional[str]:
    global db
    assert db is not None

    def _sync():
        rows = db.getFrom(
            f"SELECT translated FROM {TABLE} WHERE cache_key=%s LIMIT 1",
            (cache_key,),
            as_dict=True
        )
        if rows:
            return rows[0].get("translated")
        return None

    return await asyncio.to_thread(_sync)


async def _db_put(cache_key: str, translated: str) -> None:
    global db
    assert db is not None

    def _sync():
        return db.executeTo(
            f"""
            INSERT INTO {TABLE} (cache_key, translated)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE translated=VALUES(translated)
            """,
            (cache_key, translated)
        )

    ok = await asyncio.to_thread(_sync)
    if not ok:
        raise RuntimeError("DB write failed")



async def _call_libretranslate(text: str, source: Optional[str], target: str, fmt: str) -> str:
    assert http is not None
    payload = {
        "q": text,
        "source": source or "auto",
        "target": target,
        "format": fmt,
    }
    r = await http.post(f"{LT_URL}/translate", json=payload, timeout=120.0)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"LibreTranslate error: {r.status_code} {r.text}")
    data = r.json()
    return data.get("translatedText", "")


async def _worker_loop(worker_id: int):
    while True:
        job = await queue.get()
        cache_key, text, source, target, fmt = job
        t0 = time.time()
        try:
            async with sem:
                translated = await _call_libretranslate(text, source, target, fmt)

            await _db_put(cache_key=cache_key, translated=translated)

            # spełnij wszystkie oczekujące futures dla tego cache_key
            async with inflight_lock:
                fut = inflight.pop(cache_key, None)
            if fut and not fut.done():
                fut.set_result(translated)

        except Exception as e:
            async with inflight_lock:
                fut = inflight.pop(cache_key, None)
            if fut and not fut.done():
                fut.set_exception(e)
        finally:
            queue.task_done()


@app.on_event("startup")
async def startup():
    global db, http
    db = MySQLModel(permanent_connection=True)  # Twój wrapper trzyma global conn 
    http = httpx.AsyncClient()

    for i in range(CONCURRENCY):
        asyncio.create_task(_worker_loop(i))


@app.on_event("shutdown")
async def shutdown():
    global db, http
    if http:
        await http.aclose()
    if db:
        db.close_connection()
        db = None



@app.get("/health")
async def health():
    return {"ok": True, "queue": queue.qsize(), "concurrency": CONCURRENCY}


@app.post("/translate", response_model=TranslateOut)
async def translate(inp: TranslateIn):
    t0 = time.time()
    text = inp.text
    source = inp.source
    target = inp.target.strip().lower()
    fmt = inp.format.strip().lower()

    if fmt not in ("text", "html"):
        raise HTTPException(status_code=400, detail="format must be 'text' or 'html'")

    ck = _cache_key(text, source, target, fmt)

    # 1) cache hit
    cached = await _db_get(ck)
    if cached is not None:
        return TranslateOut(text=cached, cached=True, cache_key=ck, ms=int((time.time() - t0) * 1000))

    # 2) in-flight dedupe (jeśli już ktoś to tłumaczy, dołączamy się)
    async with inflight_lock:
        fut = inflight.get(ck)
        if fut is None:
            fut = asyncio.get_running_loop().create_future()
            inflight[ck] = fut
            # wrzucamy do kolejki TYLKO raz
            try:
                queue.put_nowait((ck, text, source, target, fmt))
            except asyncio.QueueFull:
                inflight.pop(ck, None)
                raise HTTPException(status_code=503, detail="Queue is full")

    # 3) czekamy na wynik z kolejki (kontrolowany throughput)
    try:
        translated = await asyncio.wait_for(fut, timeout=WAIT_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Translation timeout (queue busy)")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Translation failed: {repr(e)}")

    return TranslateOut(text=translated, cached=False, cache_key=ck, ms=int((time.time() - t0) * 1000))


@app.post("/translate_batch")
async def translate_batch(inp: BatchIn):
    # batch robi po kolei, ale cache + dedupe + kolejka zrobią swoje
    out = []
    for t in inp.texts:
        if not t:
            out.append("")
            continue
        res = await translate(TranslateIn(text=t, source=inp.source, target=inp.target, format=inp.format))
        out.append(res.text)
    return {"texts": out}
