import os
import json
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from openai import OpenAI
from dotenv import load_dotenv
import httpx
try:
    from .agents.orchestrator import run_all_parallel, save_to_airtable
except Exception:
    run_all_parallel = None  # type: ignore
    save_to_airtable = None  # type: ignore

# Lazy-load chat agent so we can surface import errors and try multiple paths
chat_respond = None  # type: ignore
ChatTurn = None  # type: ignore
CHAT_IMPORT_ERROR: Optional[str] = None

def _load_chat_impl() -> None:
    global chat_respond, ChatTurn, CHAT_IMPORT_ERROR
    if chat_respond is not None and ChatTurn is not None:
        return
    try:
        # Primary: relative import
        from .agents_sdk.orchestrator import chat_respond as cr, ChatTurn as CT  # type: ignore
        chat_respond, ChatTurn = cr, CT
        CHAT_IMPORT_ERROR = None
        return
    except Exception as e1:
        CHAT_IMPORT_ERROR = f"relative import failed: {e1}"
    try:
        # Fallback: absolute import (if server is a package on sys.path)
        from server.agents_sdk.orchestrator import chat_respond as cr, ChatTurn as CT  # type: ignore
        chat_respond, ChatTurn = cr, CT
        CHAT_IMPORT_ERROR = None
        return
    except Exception as e2:
        CHAT_IMPORT_ERROR = (CHAT_IMPORT_ERROR or "") + f"; absolute import failed: {e2}"
from urllib.parse import quote as urlquote

load_dotenv(override=True)

app = FastAPI(title="Markit Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

class BriefRequest(BaseModel):
    email: EmailStr
class ChatStartRequest(BaseModel):
    email: EmailStr


class ChatSendRequest(BaseModel):
    session_id: str
    email: EmailStr
    message: str


# In-memory chat sessions (basic; can swap to Redis later)
CHAT_SESSIONS: Dict[str, List[Dict[str, str]]] = {}


def _airtable_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

async def _airtable_find_by_email(client_http: httpx.AsyncClient, base_id: str, table: str, api_key: str, email: str) -> Optional[Dict[str, Any]]:
    # Many Airtable endpoints require field NAMES in filterByFormula. Try name first, then ID fallback.
    formulas = [
        f"LOWER({{Email}})='{email.lower()}'",
        f"LOWER({{fldXhVuckpHBhWJOX}})='{email.lower()}'",
    ]
    url = f"https://api.airtable.com/v0/{base_id}/{urlquote(table, safe='')}"
    last_err: Optional[Exception] = None
    for f in formulas:
        try:
            r = await client_http.get(url, headers=_airtable_headers(api_key), params={
                "maxRecords": 1,
                "filterByFormula": f,
            })
            r.raise_for_status()
            data = r.json()
            recs = data.get("records", [])
            if recs:
                return recs[0]
        except Exception as e:
            last_err = e
            continue
    if last_err:
        # swallow; higher level will record an error status if needed
        pass
    return None

async def _airtable_create_record(client_http: httpx.AsyncClient, base_id: str, table: str, api_key: str, email: str) -> Dict[str, Any]:
    url = f"https://api.airtable.com/v0/{base_id}/{urlquote(table, safe='')}"
    payload = {"records": [{"fields": {"fldXhVuckpHBhWJOX": email}}]}
    r = await client_http.post(url, headers=_airtable_headers(api_key), json=payload)
    r.raise_for_status()
    return r.json()["records"][0]

async def _airtable_update_record(client_http: httpx.AsyncClient, base_id: str, table: str, api_key: str, record_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    url = f"https://api.airtable.com/v0/{base_id}/{urlquote(table, safe='')}"
    payload = {"records": [{"id": record_id, "fields": fields}]}
    r = await client_http.patch(url, headers=_airtable_headers(api_key), json=payload)
    r.raise_for_status()
    return r.json()

async def _run_generation_and_update(email: str, api_key: str, airtable_api_key: str, airtable_base_id: str, airtable_table: str, record_id: str) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    developer_text = (
        "<ROLE> You are a marketing research analyst. Given an email with a business URL, separate the URL and use the web search tool to produce a concise, specific, marketing-ready JSON brief for PR, UGC, and creator workflows. </ROLE>\n"
        "<PRINCIPLES>\nAlways use web search; no prior knowledge.\nUse primary sources first, then secondary (≤24 months).\nNo fluff; only specific, actionable facts.\nIf unverifiable or <70% confidence, return null.\nOutput exactly in schema below. </PRINCIPLES>\n"
        "<PROCESS>\nConfirm correct company from URL.\n"
        "general_info: business_name, one_liner, website.\n"
        "products: For each top product — name, description, pricing_summary, key_features, pain_points_solved, target_use_cases.\n"
        "icp: 3–5 sentence vivid persona of the perfect customer for top product(s).\n"
        "competitors: Direct only — name, url, why_competes.\n"
        "topics_keywords:\nFrom audience perspective, pick influencer topics/niches most aligned with products + ICP.\n"
        "Exactly 10 keywords, 1–2 words each (1 word preferred).\n"
        "Avoid obscure terms unless ICP uses them often.\n"
        "No grouping, flat list. </PROCESS>\n"
        "<OUTPUT_SCHEMA>\njson: {   \"general_info\": {     \"business_name\": \"\",     \"one_liner\": \"\",     \"website\": \"\"   },   \"products\": [     {       \"name\": \"\",       \"description\": \"\",       \"pricing_summary\": \"\",       \"key_features\": [],       \"pain_points_solved\": [],       \"target_use_cases\": []     }   ],   \"icp\": \"\",   \"competitors\": [     {       \"name\": \"\",       \"url\": \"\",       \"why_competes\": \"\"     }   ],   \"topics_keywords\": [\"\", \"\", \"\", \"\", \"\", \"\", \"\", \"\", \"\", \"\"] } \n"
        "<STYLE>\nKeep all text short, specific, and marketing-useful.\nICP must be vivid and realistic (job title, goals, challenges, buying behavior).\nInclude pain points inside each product.\nAvoid corporate trivia. </STYLE>\n"
        "<FAILSAFE> If a required field is unverifiable or <70% confident, return null. If you are unsure about the company, return \"null\" and nothing else"
    )

    resp = client.responses.create(
        model="gpt-5",
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": developer_text}]},
            {"role": "user", "content": [{"type": "input_text", "text": email}]},
        ],
        text={"format": {"type": "text"}, "verbosity": "medium"},
        reasoning={"effort": "high", "summary": "detailed"},
        tools=[{"type": "web_search_preview", "user_location": {"type": "approximate", "country": "US"}, "search_context_size": "medium"}],
        store=True,
    )

    output_text = getattr(resp, "output_text", None)
    data: Any = output_text
    try:
        if isinstance(output_text, str):
            data = json.loads(output_text)
    except Exception:
        pass

    # Update Airtable with fields
    fields_update: Dict[str, Any] = {
        "fldNLJlEqVwvOg100": json.dumps(data) if not isinstance(data, str) else (data or "")
    }
    try:
        if isinstance(data, dict):
            gi = data.get("general_info", {}) if isinstance(data.get("general_info", {}), dict) else {}
            if gi.get("business_name"): fields_update["fldsrAZbfzPGLP6F8"] = str(gi.get("business_name"))
            if gi.get("one_liner"): fields_update["fldkI28kyg7gaiz2g"] = str(gi.get("one_liner"))
            if data.get("icp"): fields_update["flda7vhrHp4CuyxdC"] = str(data.get("icp"))
            if isinstance(data.get("topics_keywords"), list):
                fields_update["fldwRfzjs6xt5Vqit"] = ", ".join([str(k) for k in data.get("topics_keywords")])
    except Exception:
        pass

    async with httpx.AsyncClient(timeout=30) as http_client:
        await _airtable_update_record(http_client, airtable_base_id, airtable_table, airtable_api_key, record_id, fields_update)

    # Run 4 agents in parallel and save consolidated output
    try:
        consolidated = await run_all_parallel(api_key, email)
        await save_to_airtable(airtable_base_id, airtable_table, airtable_api_key, record_id, {"orchestration": consolidated})
    except Exception:
        pass

    return {"data": data, "raw": output_text, "record_id": record_id}


@app.post("/api/brief")
async def create_brief(req: BriefRequest, background: BackgroundTasks, wait: bool = True) -> Dict[str, Any]:
    # Reload env and construct client with current key each request
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")
    client = OpenAI(api_key=api_key)
    airtable_api_key = os.getenv("AIRTABLE_API_KEY")
    airtable_base_id = os.getenv("AIRTABLE_BASE_ID")
    # Sanitize table env (strip whitespace or stray '%' from copy/paste)
    _tbl = os.getenv("AIRTABLE_TABLE", "OAI Hackathon") or "OAI Hackathon"
    airtable_table = _tbl.strip().rstrip('%')

    if not airtable_api_key or not airtable_base_id:
        # Continue without Airtable, but note missing config
        airtable_enabled = False
    else:
        airtable_enabled = True
    airtable_status: Dict[str, Any] = {"enabled": airtable_enabled}

    developer_text = (
        "<ROLE> You are a marketing research analyst. Given an email with a business URL, separate the URL and use the web search tool to produce a concise, specific, marketing-ready JSON brief for PR, UGC, and creator workflows. </ROLE>\n"
        "<PRINCIPLES>\nAlways use web search; no prior knowledge.\nUse primary sources first, then secondary (≤24 months).\nNo fluff; only specific, actionable facts.\nIf unverifiable or <70% confidence, return null.\nOutput exactly in schema below. </PRINCIPLES>\n"
        "<PROCESS>\nConfirm correct company from URL.\n"
        "general_info: business_name, one_liner, website.\n"
        "products: For each top product — name, description, pricing_summary, key_features, pain_points_solved, target_use_cases.\n"
        "icp: 3–5 sentence vivid persona of the perfect customer for top product(s).\n"
        "competitors: Direct only — name, url, why_competes.\n"
        "topics_keywords:\nFrom audience perspective, pick influencer topics/niches most aligned with products + ICP.\n"
        "Exactly 10 keywords, 1–2 words each (1 word preferred).\n"
        "Avoid obscure terms unless ICP uses them often.\n"
        "No grouping, flat list. </PROCESS>\n"
        "<OUTPUT_SCHEMA>\njson: {   \"general_info\": {     \"business_name\": \"\",     \"one_liner\": \"\",     \"website\": \"\"   },   \"products\": [     {       \"name\": \"\",       \"description\": \"\",       \"pricing_summary\": \"\",       \"key_features\": [],       \"pain_points_solved\": [],       \"target_use_cases\": []     }   ],   \"icp\": \"\",   \"competitors\": [     {       \"name\": \"\",       \"url\": \"\",       \"why_competes\": \"\"     }   ],   \"topics_keywords\": [\"\", \"\", \"\", \"\", \"\", \"\", \"\", \"\", \"\", \"\"] } \n"
        "<STYLE>\nKeep all text short, specific, and marketing-useful.\nICP must be vivid and realistic (job title, goals, challenges, buying behavior).\nInclude pain points inside each product.\nAvoid corporate trivia. </STYLE>\n"
        "<FAILSAFE> If a required field is unverifiable or <70% confident, return null. If you are unsure about the company, return \"null\" and nothing else"
    )

    # Airtable: find or create
    existing_record = None
    created_record = None
    if airtable_enabled:
        try:
            async with httpx.AsyncClient(timeout=30) as http_client:
                existing_record = await _airtable_find_by_email(http_client, airtable_base_id, airtable_table, airtable_api_key, str(req.email))
                airtable_status["checked"] = True
                if existing_record:
                    airtable_status["existing_record_id"] = existing_record.get("id")
                    return {
                        "ok": True,
                        "mode": "existing",
                        "record_id": existing_record.get("id"),
                        "fields": existing_record.get("fields", {}),
                        "airtable": airtable_status,
                    }
                created_record = await _airtable_create_record(http_client, airtable_base_id, airtable_table, airtable_api_key, str(req.email))
                airtable_status["created_record_id"] = created_record.get("id")
        except httpx.HTTPError as e:
            airtable_status["error"] = f"HTTPError: {e}"
        except Exception as e:
            airtable_status["error"] = f"Error: {e}"

    # Run generation either inline (wait=True) or schedule in background (wait=False)
    if created_record and airtable_enabled:
        if wait:
            try:
                result = await _run_generation_and_update(str(req.email), api_key, airtable_api_key, airtable_base_id, airtable_table, created_record["id"])
                airtable_status["updated"] = True
                return {
                    "ok": True,
                    "mode": "created",
                    "email": str(req.email),
                    "record_id": created_record.get("id"),
                    "data": result.get("data"),
                    "raw": result.get("raw"),
                    "airtable": airtable_status,
                }
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"OpenAI/Airtable processing failed: {e}")
        else:
            background.add_task(_run_generation_and_update, str(req.email), api_key, airtable_api_key, airtable_base_id, airtable_table, created_record["id"]) 
            return {
                "ok": True,
                "mode": "queued",
                "email": str(req.email),
                "record_id": created_record.get("id"),
                "airtable": airtable_status,
            }
    # If Airtable disabled, still run and return data when wait=True
    if wait:
        try:
            result = await _run_generation_and_update(str(req.email), api_key, airtable_api_key or "", airtable_base_id or "", airtable_table or "OAI Hackathon", record_id="")
            return {"ok": True, "mode": "created", "email": str(req.email), "record_id": None, "data": result.get("data"), "raw": result.get("raw"), "airtable": airtable_status}
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"OpenAI processing failed: {e}")
    else:
        background.add_task(_run_generation_and_update, str(req.email), api_key, airtable_api_key or "", airtable_base_id or "", airtable_table or "OAI Hackathon", record_id="")
        return {"ok": True, "mode": "queued", "email": str(req.email), "record_id": None, "airtable": airtable_status}


@app.post("/api/chat/start")
async def chat_start(req: ChatStartRequest) -> Dict[str, Any]:
    _load_chat_impl()
    session_id = os.urandom(8).hex()
    CHAT_SESSIONS[session_id] = []

    # Try to fetch the user's prior research brief from Airtable for richer context
    load_dotenv(override=True)
    airtable_api_key = os.getenv("AIRTABLE_API_KEY")
    airtable_base_id = os.getenv("AIRTABLE_BASE_ID")
    _tbl2 = os.getenv("AIRTABLE_TABLE", "OAI Hackathon") or "OAI Hackathon"
    airtable_table = _tbl2.strip().rstrip('%')
    brief: Optional[Dict[str, Any]] = None
    record_id: Optional[str] = None
    airtable_meta: Dict[str, Any] = {"enabled": bool(airtable_api_key and airtable_base_id)}
    if airtable_api_key and airtable_base_id:
        try:
            async with httpx.AsyncClient(timeout=20) as http_client:
                rec = await _airtable_find_by_email(http_client, airtable_base_id, airtable_table, airtable_api_key, str(req.email))
                if rec:
                    record_id = rec.get("id")
                    fields = rec.get("fields", {})
                    # Primary field where we stored the JSON brief
                    raw = fields.get("fldNLJlEqVwvOg100") or fields.get("Full Response") or ""
                    if isinstance(raw, str) and raw:
                        try:
                            parsed = json.loads(raw)
                            if isinstance(parsed, dict):
                                brief = parsed
                        except Exception:
                            # Fallback: sometimes Airtable stores JSON-like text; try heuristics
                            brief = None
                    # Secondary heuristic: if any field looks like the schema
                    if not brief:
                        for v in fields.values():
                            if isinstance(v, str) and '"general_info"' in v and '"products"' in v:
                                try:
                                    parsed2 = json.loads(v)
                                    if isinstance(parsed2, dict):
                                        brief = parsed2
                                        break
                                except Exception:
                                    continue
            airtable_meta["record_id"] = record_id
            airtable_meta["has_brief"] = bool(brief)
        except Exception as e:
            airtable_meta["error"] = str(e)

    # Build hidden greeting turn for the orchestrator
    email_str = str(req.email)
    domain = email_str.split("@", 1)[1] if "@" in email_str else ""
    hidden_user_prompt = (
        f"you are greeting {email_str} who works at {domain}. below is their business information, in less than 240 chars greet them, let them know you know about their company and reiterate a one-liner to show you do, and ask what marketing task they'd like to execute next (open ended). Be friendly and professional.\n" 
        + ("BUSINESS_INFO:\n" + json.dumps(brief) + "\n" if brief else "BUSINESS_INFO:\n{}\n")
    )

    # If chat agent is available, get the first assistant reply now (does not show the hidden user turn)
    first_reply = None
    meta: Dict[str, Any] = {"airtable": airtable_meta}
    if chat_respond is not None and ChatTurn is not None:
        try:
            init_history: List[Dict[str, str]] = [{"role": "user", "content": hidden_user_prompt}]
            turns: List[ChatTurn] = [ChatTurn(**t) for t in init_history]  # type: ignore[arg-type]
            reply, m = await chat_respond(email_str, turns)
            first_reply = reply or ""
            meta.update(m or {})
            # Persist both hidden user turn and assistant reply in the server session history
            CHAT_SESSIONS[session_id].extend(init_history)
            if first_reply:
                CHAT_SESSIONS[session_id].append({"role": "assistant", "content": first_reply})
        except Exception as e:
            meta["error"] = f"prefetch_greeting_failed: {e}"

    return {
        "ok": True,
        "session_id": session_id,
        "agent_ready": chat_respond is not None,
        "import_error": CHAT_IMPORT_ERROR,
        "first_reply": first_reply,
        "meta": meta,
    }


@app.post("/api/chat/send")
async def chat_send(req: ChatSendRequest) -> Dict[str, Any]:
    _load_chat_impl()
    if chat_respond is None:
        raise HTTPException(status_code=500, detail=f"Chat agent not available: {CHAT_IMPORT_ERROR}")
    hist = CHAT_SESSIONS.get(req.session_id)
    if hist is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    # Append user turn
    hist.append({"role": "user", "content": req.message})

    # Build pydantic ChatTurn list and get agent reply
    turns: List[ChatTurn] = [ChatTurn(**t) for t in hist]  # type: ignore[arg-type]
    reply, meta = await chat_respond(str(req.email), turns)
    # Surface raw request/response debug into API for the UI debug console
    meta["_debug"] = {
        "history": hist,
        "email": str(req.email),
    }

    # Append assistant turn
    hist.append({"role": "assistant", "content": reply})
    return {"ok": True, "reply": reply, "meta": meta}


@app.get("/api/chat/status")
async def chat_status(session_id: str) -> Dict[str, Any]:
    """Return the current session history and the last assistant message, if any."""
    hist = CHAT_SESSIONS.get(session_id)
    if hist is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    last_assistant = None
    for turn in reversed(hist):
        if turn.get("role") == "assistant":
            last_assistant = turn.get("content", "")
            break
    return {"ok": True, "session_id": session_id, "history": hist, "last_assistant": last_assistant}
