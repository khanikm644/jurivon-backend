"""
JURIVON AI - Version 2.0
Private Legal AI Platform for Law Firms
FastAPI Backend - Complete Implementation
"""

import os, re, json, asyncio, logging, traceback
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# ── External libs ──────────────────────────────────────────
import httpx
import pdfplumber
import io
from openai import AsyncOpenAI
from supabase import create_client, Client
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger("jurivon")

# ── Config ─────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL    = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY", "")
ENVIRONMENT     = os.getenv("ENVIRONMENT", "production")
FRONTEND_URL    = os.getenv("FRONTEND_URL", "https://jurivon-frontend.vercel.app")

ALLOWED_ORIGINS = [
    "https://jurivon-frontend.vercel.app",
    "https://web-production-7ecc0.up.railway.app",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8080",
    "null",
]

ALLOWED_FILE_TYPES = {
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ── Clients ────────────────────────────────────────────────
openai_client: AsyncOpenAI = None
supabase: Client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global openai_client, supabase
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info(f"Jurivon v2.0 started — environment: {ENVIRONMENT}")
    yield
    logger.info("Jurivon shutting down")

# ── App ────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Jurivon AI", version="2.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── Global error handler ───────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={
        "success": False,
        "error": "internal_error",
        "message": "Something went wrong. Our team has been notified. Please try again.",
        "data": None
    })
# ── Health checks (Railway probe + frontend status) ────────
async def health_check():
    return {
        "status": "ok",
        "version": "2.0.0",
        "environment": ENVIRONMENT,
        "supabase": supabase is not None,
        "openai": bool(OPENAI_API_KEY)
    }
# ── Response helpers ───────────────────────────────────────
def ok(data, message="OK"):
    return {"success": True, "data": data, "error": None, "message": message}

def err(message: str, code: int = 400):
    return JSONResponse(status_code=code, content={
        "success": False, "data": None,
        "error": message, "message": message
    })

# ── File validation ────────────────────────────────────────
async def validate_file(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(400, "Only PDF, TXT, or DOCX files are allowed.")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large. Maximum size is 10MB.")
    return content

# ── Text extraction ────────────────────────────────────────
def extract_text(content: bytes, filename: str) -> str:
    try:
        if filename.lower().endswith(".pdf"):
            text = ""
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
            return text.strip()
        else:
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    return content.decode(enc).strip()
                except UnicodeDecodeError:
                    continue
            return content.decode("utf-8", errors="replace").strip()
    except Exception as e:
        logger.error(f"Text extraction error: {e}")
        return ""

# ── OpenAI with retry ──────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)
async def call_ai(messages: list, max_tokens: int = 2000, temperature: float = 0) -> str:
    r = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=45
    )
    return r.choices[0].message.content

# ── Audit log ──────────────────────────────────────────────
async def log_action(firm_id: str, user: str, action: str, subject: str,
                     result: str, workspace_id: str = None, full_content: dict = None):
    if not supabase:
        return
    try:
        record = {
            "firm_id": firm_id,
            "user_name": user,
            "action": action,
            "subject": subject[:500],
            "result": result[:200],
            "session_id": f"v2-{datetime.utcnow().strftime('%Y%m%d')}",
            "created_at": datetime.utcnow().isoformat()
        }
        supabase.table("audit_log").insert(record).execute()
        if full_content:
            history = {
                "firm_id": firm_id,
                "user_name": user,
                "feature": action,
                "input_summary": subject[:500],
                "full_content": full_content,
                "workspace_id": workspace_id,
                "created_at": datetime.utcnow().isoformat()
            }
            supabase.table("interaction_history").insert(history).execute()
    except Exception as e:
        logger.warning(f"Audit log failed: {e}")

# ══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════
class ConflictRequest(BaseModel):
    matter_description: str = Field(..., min_length=5, max_length=5000)
    firm_id: str = Field(default="default", max_length=100)
    user_name: str = Field(default="User", max_length=100)
    workspace_id: Optional[str] = None

    @validator("matter_description")
    def sanitize(cls, v):
        return re.sub(r'<[^>]+>', '', v).strip()

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000)
    jurisdiction: str = Field(..., min_length=2, max_length=20)
    firm_id: str = Field(default="default")
    user_name: str = Field(default="User")
    workspace_id: Optional[str] = None

    @validator("jurisdiction")
    def validate_jurisdiction(cls, v):
        allowed = ["UK","Italy","EU","UAE","Pakistan","US","Germany","France","Global"]
        if v not in allowed:
            raise ValueError(f"Jurisdiction must be one of: {', '.join(allowed)}")
        return v

class DraftRequest(BaseModel):
    doc_type: str = Field(..., min_length=2, max_length=100)
    party_a: str = Field(..., min_length=1, max_length=200)
    party_b: str = Field(..., min_length=1, max_length=200)
    jurisdiction: str = Field(default="UK", max_length=50)
    key_terms: str = Field(default="", max_length=2000)
    firm_id: str = Field(default="default")
    user_name: str = Field(default="User")
    workspace_id: Optional[str] = None

class CitationRequest(BaseModel):
    citation: str = Field(..., min_length=3, max_length=500)
    jurisdiction: str = Field(default="UK")
    firm_id: str = Field(default="default")

class OdvRequest(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=200)
    company_type: str = Field(default="S.r.l.")
    lawyer_name: str = Field(..., min_length=1, max_length=200)
    lawyer_bar_number: str = Field(default="", max_length=50)
    matter_description: str = Field(..., min_length=10, max_length=2000)
    conflict_result: str = Field(default="CLEAR")
    firm_id: str = Field(default="default")
    user_name: str = Field(default="User")

class WorkspaceCreate(BaseModel):
    client_name: str = Field(..., min_length=1, max_length=200)
    matter_ref: Optional[str] = Field(None, max_length=100)
    matter_type: Optional[str] = Field(None, max_length=100)
    jurisdiction: str = Field(default="UK")
    lead_partner: Optional[str] = Field(None, max_length=100)
    firm_id: str = Field(default="default")

class WorkspaceItemCreate(BaseModel):
    workspace_id: str
    item_type: str = Field(..., max_length=50)
    title: str = Field(..., min_length=1, max_length=200)
    content: dict
    firm_id: str = Field(default="default")

class CommentCreate(BaseModel):
    workspace_id: str
    author: str = Field(..., min_length=1, max_length=100)
    comment: str = Field(..., min_length=1, max_length=2000)
    firm_id: str = Field(default="default")

class MatterRecord(BaseModel):
    matter_ref: str = Field(..., min_length=1, max_length=100)
    client: str = Field(..., min_length=1, max_length=200)
    counterparty: Optional[str] = Field(None, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)
    status: str = Field(default="OPEN")
    lead_partner: Optional[str] = Field(None, max_length=100)
    practice_area: Optional[str] = Field(None, max_length=100)
    firm_id: str = Field(default="default")

# ══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════
@app.get("/health")
async def health():
    db_ok = False
    if supabase:
        try:
            supabase.table("firm_settings").select("id").limit(1).execute()
            db_ok = True
        except:
            pass
    return {
        "status": "ok" if db_ok else "degraded",
        "version": "2.0.0",
        "database": "connected" if db_ok else "unavailable",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/")
async def root():
    return {"name": "Jurivon AI", "version": "2.0.0", "status": "running"}

# ══════════════════════════════════════════════════════════════
# FEATURE 1 — CONFLICT CHECK
# ══════════════════════════════════════════════════════════════
@app.post("/api/v1/conflict-check")
@limiter.limit("20/minute")
async def conflict_check(request: Request, data: ConflictRequest):
    matters = []
    if supabase:
        try:
            r = supabase.table("matters").select("*").eq("firm_id", data.firm_id).execute()
            matters = r.data or []
        except Exception as e:
            logger.warning(f"Matter DB read failed: {e}")

    matters_text = ""
    if matters:
        for m in matters:
            matters_text += (f"REF: {m.get('matter_ref','')} | "
                f"CLIENT: {m.get('client','')} | "
                f"COUNTERPARTY: {m.get('counterparty','')} | "
                f"DESC: {m.get('description','')[:200]}\n")
    else:
        matters_text = "No matters found in database."

    system = """You are a senior legal conflicts officer. Analyse the new matter against the existing matter database.

Return a structured response in this exact format:
CONFLICT RESULT: [HIGH RISK / MEDIUM RISK / LOW RISK / CLEAR]
CONFIDENCE: [percentage]
CONFLICTS FOUND: [list each conflict with matter reference, or 'None']
ENTITIES IDENTIFIED: [list all entities extracted from new matter]
RECOMMENDATION: [1-2 sentences on what to do]
EXPLANATION: [brief explanation of your assessment]"""

    user = f"""NEW MATTER:
{data.matter_description}

EXISTING MATTER DATABASE:
{matters_text[:6000]}

Check for conflicts. Extract all entity names from the new matter and compare against all clients and counterparties in the database."""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ], max_tokens=1000)
        risk = "HIGH" if "HIGH RISK" in result else "MEDIUM" if "MEDIUM RISK" in result else "LOW" if "LOW RISK" in result else "CLEAR"
        await log_action(data.firm_id, data.user_name, "conflict_check",
                        data.matter_description[:200], risk, data.workspace_id,
                        {"query": data.matter_description, "result": result, "risk_level": risk})
        return ok({"result": result, "risk_level": risk, "matters_checked": len(matters)})
    except Exception as e:
        logger.error(f"Conflict check error: {e}")
        return err("Conflict check failed. Please try again.")

# ══════════════════════════════════════════════════════════════
# FEATURE 2 — DOCUMENT Q&A
# ══════════════════════════════════════════════════════════════
@app.post("/api/v1/document-qa")
@limiter.limit("15/minute")
async def document_qa(
    request: Request,
    file: UploadFile = File(...),
    question: str = Form(...),
    firm_id: str = Form(default="default"),
    user_name: str = Form(default="User"),
    workspace_id: str = Form(default="")
):
    content = await validate_file(file)
    text = extract_text(content, file.filename)
    if not text:
        return err("Could not extract text from the document.")
    if len(question.strip()) < 3:
        return err("Please enter a valid question.")

    system = """You are a senior legal analyst reviewing a document on behalf of a law firm.

Answer the question based ONLY on the document provided.
Format:
DIRECT ANSWER: [clear answer in 1-3 sentences]
RELEVANT CLAUSE: [exact quote from document, max 100 words]
LOCATION: [page/section if identifiable]
CONFIDENCE: [HIGH/MEDIUM/LOW]
CAVEAT: [any important caveats or limitations]"""

    user_msg = f"""DOCUMENT ({file.filename}):
{text[:12000]}

QUESTION: {question}"""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ], max_tokens=1500)
        await log_action(firm_id, user_name, "document_qa",
                        f"{file.filename}: {question[:100]}", "Answered",
                        workspace_id or None, {"filename": file.filename,
                        "question": question, "answer": result})
        return ok({"result": result, "filename": file.filename})
    except Exception as e:
        return err("Document Q&A failed. Please try again.")

# ══════════════════════════════════════════════════════════════
# FEATURE 3 — CONTRACT REVIEW
# ══════════════════════════════════════════════════════════════
@app.post("/api/v1/contract-review")
@limiter.limit("10/minute")
async def contract_review(
    request: Request,
    file: UploadFile = File(...),
    firm_id: str = Form(default="default"),
    user_name: str = Form(default="User"),
    workspace_id: str = Form(default="")
):
    content = await validate_file(file)
    text = extract_text(content, file.filename)
    if not text:
        return err("Could not extract text from document.")

    system = """You are a senior commercial lawyer reviewing a contract for a law firm client.

Identify all issues and risks. Use this exact format for each issue:

RISK: [HIGH/MEDIUM/LOW]
CLAUSE: [clause title or number]
ISSUE: [what the problem is in plain English]
QUOTE: [relevant text from contract, max 50 words]
RECOMMENDATION: [specific action to fix it]

After all issues, add:
SUMMARY: [overall assessment in 2 sentences]
MISSING CLAUSES: [list any important missing clauses]
OVERALL RISK RATING: [HIGH/MEDIUM/LOW]"""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": f"Review this contract:\n\n{text[:14000]}"}
        ], max_tokens=2500)
        await log_action(firm_id, user_name, "contract_review",
                        file.filename, "Completed", workspace_id or None,
                        {"filename": file.filename, "review": result})
        return ok({"result": result, "filename": file.filename})
    except Exception as e:
        return err("Contract review failed. Please try again.")

# ══════════════════════════════════════════════════════════════
# FEATURE 4 — DOCUMENT SUMMARY
# ══════════════════════════════════════════════════════════════
@app.post("/api/v1/document-summary")
@limiter.limit("15/minute")
async def document_summary(
    request: Request,
    file: UploadFile = File(...),
    firm_id: str = Form(default="default"),
    user_name: str = Form(default="User"),
    workspace_id: str = Form(default="")
):
    content = await validate_file(file)
    text = extract_text(content, file.filename)
    if not text:
        return err("Could not extract text from document.")

    system = """You are a senior legal analyst. Produce a structured one-page brief of this document.

Format exactly as follows:
DOCUMENT TYPE: [type]
PARTIES: [Party A: name and role | Party B: name and role]
DATE: [execution date if present]
GOVERNING LAW: [jurisdiction]
PURPOSE: [1 sentence]
KEY OBLIGATIONS:
- Party A: [obligations]
- Party B: [obligations]
IMPORTANT DATES & DEADLINES: [list]
KEY DEFINED TERMS: [list most important ones]
RISK CLAUSES: [HIGH/MED/LOW for each flagged clause]
OVERALL SUMMARY: [2 sentences]"""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": f"Summarise:\n\n{text[:14000]}"}
        ], max_tokens=1500)
        await log_action(firm_id, user_name, "document_summary",
                        file.filename, "Completed", workspace_id or None,
                        {"filename": file.filename, "summary": result})
        return ok({"result": result, "filename": file.filename})
    except Exception as e:
        return err("Document summary failed. Please try again.")

# ══════════════════════════════════════════════════════════════
# FEATURE 5 — DRAFT WITH LEX (streaming)
# ══════════════════════════════════════════════════════════════
@app.post("/api/v1/draft-with-lex")
@limiter.limit("10/minute")
async def draft_with_lex(request: Request, data: DraftRequest):
    jurisdiction_rules = {
        "UK": "English law. Use UK legal conventions. Cite relevant statutes (Companies Act 2006, Contract Act, etc.).",
        "Italy": "Italian law. Use Italian legal conventions. Reference Codice Civile where applicable.",
        "UAE": "UAE law with DIFC/ADGM framework where applicable. Reference UAE Federal Laws.",
        "Pakistan": "Pakistani law. Reference Contract Act 1872, relevant Pakistani statutes. Use formal Pakistani legal style.",
        "EU": "EU law. Reference relevant EU Directives and Regulations.",
        "US": "US law. Specify governing state. Use US legal conventions.",
        "Germany": "German law (BGB). Use German legal conventions.",
    }
    jur_rule = jurisdiction_rules.get(data.jurisdiction,
                                       f"{data.jurisdiction} law and legal conventions.")

    system = f"""You are Lex — a senior legal drafting AI for Jurivon. You draft professional legal documents.

Jurisdiction rules: {jur_rule}

Draft rules:
1. Use professional legal language appropriate for the jurisdiction
2. Mark every clause requiring specific partner review with [REVIEW REQUIRED: reason]
3. Use [PARTY A] and [PARTY B] as placeholders where names should be inserted
4. Mark uncertain or high-risk clauses with [LEGAL ADVICE REQUIRED]
5. Include all standard clauses for this document type
6. Add a brief drafter's note at the end explaining any key choices made

Always start with the document title and date placeholder."""

    user_msg = f"""Draft a {data.doc_type}

Party A: {data.party_a}
Party B: {data.party_b}
Jurisdiction: {data.jurisdiction}
Additional terms: {data.key_terms if data.key_terms else 'Standard terms apply'}"""

    async def stream_draft():
        try:
            stream = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg}
                ],
                stream=True,
                max_tokens=3000,
                temperature=0.1
            )
            full_text = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_text += delta
                    yield f"data: {json.dumps({'chunk': delta})}\n\n"
            await log_action(data.firm_id, data.user_name, "draft_with_lex",
                           f"{data.doc_type} — {data.party_a} / {data.party_b}",
                           "Drafted", data.workspace_id,
                           {"doc_type": data.doc_type, "draft": full_text[:5000]})
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream_draft(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ══════════════════════════════════════════════════════════════
# FEATURE 6 — DUE DILIGENCE
# ══════════════════════════════════════════════════════════════
@app.post("/api/v1/due-diligence")
@limiter.limit("5/minute")
async def due_diligence(
    request: Request,
    files: List[UploadFile] = File(...),
    deal_type: str = Form(default="M&A"),
    firm_id: str = Form(default="default"),
    user_name: str = Form(default="User"),
    workspace_id: str = Form(default="")
):
    if len(files) > 10:
        return err("Maximum 10 documents per due diligence request.")

    combined = ""
    filenames = []
    for f in files:
        content = await validate_file(f)
        text = extract_text(content, f.filename)
        combined += f"\n\n=== DOCUMENT: {f.filename} ===\n{text[:3000]}"
        filenames.append(f.filename)

    system = f"""You are a senior M&A lawyer conducting {deal_type} due diligence.

Analyse the provided documents across 5 categories. For each:
- List findings as HIGH / MEDIUM / LOW risk
- Cite which document each finding comes from
- Give specific recommendation

FORMAT:

1. CORPORATE & OWNERSHIP
[findings with risk levels and source documents]

2. EMPLOYMENT & HR
[findings with risk levels and source documents]

3. INTELLECTUAL PROPERTY
[findings with risk levels and source documents]

4. LITIGATION & DISPUTES
[findings with risk levels and source documents]

5. REGULATORY & COMPLIANCE
[findings with risk levels and source documents]

OVERALL RISK SUMMARY:
Red Flags (HIGH): [count and list]
Amber Flags (MEDIUM): [count and list]
Green (LOW/CLEAR): [count and list]
RECOMMENDATION: [proceed / proceed with conditions / do not proceed]"""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": f"Documents: {', '.join(filenames)}\n\n{combined[:15000]}"}
        ], max_tokens=3000)
        await log_action(firm_id, user_name, "due_diligence",
                        f"{deal_type}: {', '.join(filenames)}", "Completed",
                        workspace_id or None, {"files": filenames, "result": result})
        return ok({"result": result, "documents_analysed": len(files)})
    except Exception as e:
        return err("Due diligence analysis failed. Please try again.")

# ══════════════════════════════════════════════════════════════
# FEATURE 7 — OdV ART. 231 CERTIFICATE (Italian firms ONLY)
# ══════════════════════════════════════════════════════════════
@app.post("/api/v1/odv-certificate")
@limiter.limit("10/minute")
async def odv_certificate(request: Request, data: OdvRequest):
    timestamp = datetime.utcnow()
    cert_ref = f"OdV-{timestamp.strftime('%Y%m%d%H%M%S')}-{data.company_name[:6].upper().replace(' ','')}"

    system = """You are drafting an Organismo di Vigilanza independence assessment
under D.Lgs. 231/2001 for Italian Bar filing.
Generate a bilingual (Italian and English) certificate.
Be formal. Use proper Italian legal language in the Italian section."""

    user_msg = f"""Generate OdV Art. 231/2001 certificate:

Company: {data.company_name} ({data.company_type})
Assessing Lawyer: Avv. {data.lawyer_name}
Bar Registration: {data.lawyer_bar_number if data.lawyer_bar_number else '[NUMBER]'}
Matter: {data.matter_description}
Conflict Check Result: {data.conflict_result}
Certificate Reference: {cert_ref}
Date: {timestamp.strftime('%d %B %Y')}

Include:
1. Italian section: formal OdV independence declaration
2. English translation
3. Independence assessment methodology
4. Recommendation (PROCEED / DO NOT PROCEED based on conflict result)
5. Signature and certification block"""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ], max_tokens=2000)
        cert_data = {
            "certificate_ref": cert_ref,
            "company": data.company_name,
            "lawyer": data.lawyer_name,
            "conflict_result": data.conflict_result,
            "date": timestamp.strftime("%d %B %Y"),
            "result": result
        }
        await log_action(data.firm_id, data.user_name, "odv_certificate",
                        f"{data.company_name} — {cert_ref}", data.conflict_result,
                        None, cert_data)
        return ok(cert_data)
    except Exception as e:
        return err("OdV certificate generation failed.")

# ══════════════════════════════════════════════════════════════
# FEATURE 8 — LEGAL RESEARCH WITH KHAN
# ══════════════════════════════════════════════════════════════

async def fetch_uk_law(query: str) -> str:
    """Fetch from legislation.gov.uk API"""
    try:
        url = "https://www.legislation.gov.uk/api/1/search"
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(url, params={"text": query, "results": "5"})
            if r.status_code == 200:
                data = r.json()
                items = data.get("results", [])[:3]
                texts = []
                for item in items:
                    link = item.get("link", "")
                    if link:
                        tr = await client.get(f"{link}/data.json", timeout=8)
                        if tr.status_code == 200:
                            jd = tr.json()
                            t = str(jd)[:2000]
                            texts.append(f"Source: {link}\n{t}")
                if texts:
                    return "\n\n---\n\n".join(texts)
    except Exception as e:
        logger.warning(f"UK law fetch failed: {e}")
    return ""

async def fetch_eu_law(query: str) -> str:
    """Fetch from EUR-Lex"""
    try:
        sparql = f"""SELECT ?title ?uri WHERE {{
          ?uri <http://purl.org/dc/elements/1.1/title> ?title .
          FILTER(CONTAINS(LCASE(STR(?title)), LCASE("{query[:50]}")))
        }} LIMIT 5"""
        url = "https://publications.europa.eu/webapi/rdf/sparql"
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(url, params={"query": sparql,
                "format": "application/sparql-results+json"}, timeout=12)
            if r.status_code == 200:
                data = r.json()
                bindings = data.get("results", {}).get("bindings", [])
                if bindings:
                    items = [f"- {b.get('title',{}).get('value','')}: {b.get('uri',{}).get('value','')}"
                             for b in bindings[:5]]
                    return "EUR-Lex results:\n" + "\n".join(items)
    except Exception as e:
        logger.warning(f"EU law fetch failed: {e}")
    return ""

async def fetch_italian_law(query: str) -> str:
    """Fetch from Normattiva"""
    try:
        url = "https://www.normattiva.it/do/atto/ricercaPerTesto"
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(url,
                params={"query": query, "typeSearch": "T"},
                headers={"User-Agent": "Jurivon Legal Research/2.0"},
                timeout=12)
            if r.status_code == 200:
                text = re.sub(r'<[^>]+>', ' ', r.text)
                text = re.sub(r'\s+', ' ', text).strip()
                return f"Normattiva results:\n{text[:3000]}"
    except Exception as e:
        logger.warning(f"Italian law fetch failed: {e}")
    return ""

async def fetch_pakistan_law_rag(query: str) -> tuple:
    """RAG search for Pakistan law from vector store"""
    if not supabase:
        return "", False
    try:
        emb = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query[:8000]
        )
        results = supabase.rpc("match_pakistan_law", {
            "query_embedding": emb.data[0].embedding,
            "match_threshold": 0.65,
            "match_count": 6
        }).execute()
        if results.data:
            chunks = []
            for r in results.data:
                chunks.append(
                    f"Source: {r.get('title','')} "
                    f"({r.get('court','')}, {r.get('year','')})\n"
                    f"Citation: {r.get('citation','Unverified')}\n"
                    f"{r.get('chunk_text','')}"
                )
            return "\n\n---\n\n".join(chunks), True
    except Exception as e:
        logger.warning(f"Pakistan law RAG failed: {e}")
    return "", False

JURISDICTION_SYSTEMS = {
    "UK": "England and Wales legal system. Common law. Key statutes: Companies Act 2006, Contract Act, GDPR UK.",
    "Italy": "Italian civil law system. Codice Civile, D.Lgs. 231/2001, Italian Bar rules. GDPR applies.",
    "EU": "European Union law. EU Directives, Regulations, CJEU case law. GDPR Regulation 2016/679.",
    "UAE": "UAE Federal law + DIFC/ADGM free zone law. Civil law with common law elements in free zones.",
    "Pakistan": "Pakistani common law system. Contract Act 1872, CPC 1908, PPC 1860, Limitation Act 1908.",
    "US": "US common law. Federal and state law. Specify governing state in answer.",
    "Germany": "German civil law (BGB). Strong codified system. GDPR applies with strict enforcement.",
    "France": "French civil law (Code Civil). GDPR applies. French Bar rules.",
    "Global": "Multi-jurisdictional. Cover key international principles.",
}

@app.post("/api/v1/legal-research-khan")
@limiter.limit("15/minute")
async def legal_research_khan(request: Request, data: ResearchRequest):
    """Legal Research with Khan — AI-powered with live law APIs"""
    jur_system = JURISDICTION_SYSTEMS.get(data.jurisdiction,
                                          f"{data.jurisdiction} legal system.")
    live_context = ""
    source_type = "AI knowledge"

    try:
        if data.jurisdiction == "UK":
            live_context = await fetch_uk_law(data.query)
            if live_context: source_type = "legislation.gov.uk (live)"
        elif data.jurisdiction in ("EU", "Germany", "France"):
            live_context = await fetch_eu_law(data.query)
            if live_context: source_type = "EUR-Lex (live)"
        elif data.jurisdiction == "Italy":
            live_context = await fetch_italian_law(data.query)
            if live_context: source_type = "Normattiva (live)"
        elif data.jurisdiction == "Pakistan":
            live_context, has_db = await fetch_pakistan_law_rag(data.query)
            if has_db: source_type = "Jurivon Pakistan Law Database"
    except Exception as e:
        logger.warning(f"Live law fetch failed: {e}")

    if live_context:
        context_prompt = f"""Retrieved texts from official sources:
{live_context[:6000]}

Answer using ONLY the above retrieved texts where possible.
Cite specific sources for every legal point."""
    else:
        context_prompt = f"""No live law texts were retrieved.
Answer from training knowledge but mark every key statement as [VERIFY AGAINST CURRENT SOURCE].
Provide official source URLs for verification."""

    system = f"""You are Khan — Jurivon's senior legal research AI. You provide expert legal research for law firms globally.

Legal system context: {jur_system}

{context_prompt}

Response format:
DIRECT ANSWER:
[clear, specific answer to the question in 2-3 sentences]

LEGAL BASIS:
[specific statutes, articles, case law with citations]

DETAILED ANALYSIS:
[comprehensive explanation]

PRACTICAL IMPLICATIONS:
[what this means for the client in practice]

IMPORTANT CAVEATS:
[limitations, recent developments, areas of uncertainty]

VERIFICATION SOURCES:
[official URLs where this can be verified]

---
Researched by: Khan | Jurivon Legal Research AI
Jurisdiction: {data.jurisdiction} | Date: {datetime.utcnow().strftime('%d %B %Y')}"""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": data.query}
        ], max_tokens=2500)
        await log_action(data.firm_id, data.user_name, "legal_research_khan",
                        f"[{data.jurisdiction}] {data.query[:200]}", "Completed",
                        data.workspace_id, {"query": data.query,
                        "jurisdiction": data.jurisdiction,
                        "source_type": source_type, "result": result})
        return ok({
            "result": result,
            "jurisdiction": data.jurisdiction,
            "source_type": source_type,
            "has_live_data": bool(live_context)
        })
    except Exception as e:
        return err("Legal research failed. Please try again.")

# ══════════════════════════════════════════════════════════════
# FEATURE 9 — CITATION VERIFY
# ══════════════════════════════════════════════════════════════
@app.post("/api/v1/citation-verify")
@limiter.limit("20/minute")
async def citation_verify(request: Request, data: CitationRequest):
    system = """You are a legal citation verification specialist. Assess whether a legal citation is real and accurate.

Return exactly this format:
VERIFICATION RESULT: [VERIFIED / LIKELY REAL / UNVERIFIABLE / SUSPICIOUS / INCORRECT]
CONFIDENCE: [percentage]
ASSESSMENT: [explanation of your assessment]
SOURCE URL: [official URL where this can be verified, or 'Not available']
WARNING: [if the citation appears fabricated or incorrect, explain why]
RECOMMENDATION: [what the lawyer should do to verify this]"""

    user_msg = f"Verify this legal citation: '{data.citation}'\nJurisdiction: {data.jurisdiction}"

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ], max_tokens=800)
        status = "VERIFIED" if "VERIFIED" in result and "UNVERIFIABLE" not in result else \
                 "SUSPICIOUS" if "SUSPICIOUS" in result or "INCORRECT" in result else "UNVERIFIABLE"
        await log_action(data.firm_id, "User", "citation_verify",
                        data.citation, status)
        return ok({"result": result, "citation": data.citation, "status": status})
    except Exception as e:
        return err("Citation verification failed.")

# ══════════════════════════════════════════════════════════════
# FEATURE 10 — REGULATORY TRACKER
# ══════════════════════════════════════════════════════════════
@app.post("/api/v1/regulatory-tracker")
@limiter.limit("10/minute")
async def regulatory_tracker(
    request: Request,
    jurisdiction: str = Form(...),
    practice_areas: str = Form(...),
    firm_id: str = Form(default="default"),
    user_name: str = Form(default="User")
):
    system = f"""You are a regulatory intelligence analyst for law firms.

Provide current regulatory developments for {jurisdiction} in the practice areas: {practice_areas}.

Format exactly:
REGULATORY TRACKER — {jurisdiction}
Practice Areas: {practice_areas}
Date: {datetime.utcnow().strftime('%d %B %Y')}

For each development:
URGENCY: [HIGH / MEDIUM / LOW]
AREA: [practice area]
DEVELOPMENT: [title]
SUMMARY: [what changed in 2-3 sentences]
EFFECTIVE DATE: [date or 'TBC']
ACTION REQUIRED: [what the firm must do]
SOURCE: [official source name and URL]

List at minimum 5 developments, more if available."""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": f"What are the latest regulatory developments for {jurisdiction} legal practice in {practice_areas}?"}
        ], max_tokens=2000)
        await log_action(firm_id, user_name, "regulatory_tracker",
                        f"{jurisdiction} — {practice_areas}", "Completed")
        return ok({"result": result, "jurisdiction": jurisdiction})
    except Exception as e:
        return err("Regulatory tracker failed.")

# ══════════════════════════════════════════════════════════════
# FEATURE 11 — MATTER DATABASE
# ══════════════════════════════════════════════════════════════
@app.get("/api/v1/matters")
async def get_matters(firm_id: str = "default"):
    if not supabase:
        return ok([])
    try:
        r = supabase.table("matters").select("*")\
            .eq("firm_id", firm_id)\
            .order("created_at", desc=True).execute()
        return ok(r.data or [])
    except Exception as e:
        return err("Failed to fetch matters.")

@app.post("/api/v1/matters")
async def add_matter(request: Request, data: MatterRecord):
    if not supabase:
        return err("Database not connected.")
    try:
        record = data.dict()
        record["created_at"] = datetime.utcnow().isoformat()
        r = supabase.table("matters").insert(record).execute()
        await log_action(data.firm_id, "System", "matter_added",
                        f"{data.matter_ref} — {data.client}", "Added")
        return ok(r.data[0] if r.data else {})
    except Exception as e:
        return err("Failed to add matter.")

@app.post("/api/v1/matters/bulk")
async def bulk_upload_matters(
    request: Request,
    file: UploadFile = File(...),
    firm_id: str = Form(default="default")
):
    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    if len(lines) > 500:
        return err("Maximum 500 matters per bulk upload.")

    matters = []
    for line in lines[1:]:  # skip header
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            matters.append({
                "firm_id": firm_id,
                "matter_ref": parts[0] if len(parts) > 0 else "REF",
                "client": parts[1] if len(parts) > 1 else "",
                "counterparty": parts[2] if len(parts) > 2 else "",
                "description": parts[3] if len(parts) > 3 else "",
                "status": parts[4] if len(parts) > 4 else "OPEN",
                "lead_partner": parts[5] if len(parts) > 5 else "",
                "practice_area": parts[6] if len(parts) > 6 else "",
                "created_at": datetime.utcnow().isoformat()
            })

    if not matters:
        return err("No valid matters found in file.")
    if supabase:
        supabase.table("matters").insert(matters).execute()
    return ok({"inserted": len(matters)})

@app.delete("/api/v1/matters/{matter_id}")
async def delete_matter(matter_id: str, firm_id: str = "default"):
    if not supabase:
        return err("Database not connected.")
    supabase.table("matters").delete().eq("id", matter_id).eq("firm_id", firm_id).execute()
    return ok({"deleted": True})

# ══════════════════════════════════════════════════════════════
# FEATURE 12 — AUDIT LOG
# ══════════════════════════════════════════════════════════════
@app.get("/api/v1/audit-log")
async def get_audit_log(firm_id: str = "default", limit: int = 100):
    if not supabase:
        return ok([])
    try:
        r = supabase.table("audit_log").select("*")\
            .eq("firm_id", firm_id)\
            .order("created_at", desc=True)\
            .limit(min(limit, 500)).execute()
        return ok(r.data or [])
    except Exception as e:
        return err("Failed to fetch audit log.")

# ══════════════════════════════════════════════════════════════
# FEATURE 13 — PRECEDENT SEARCH
# ══════════════════════════════════════════════════════════════
@app.post("/api/v1/precedent-search")
@limiter.limit("15/minute")
async def precedent_search(
    request: Request,
    query: str = Form(...),
    firm_id: str = Form(default="default"),
    user_name: str = Form(default="User")
):
    matters = []
    if supabase:
        try:
            r = supabase.table("matters").select("*").eq("firm_id", firm_id).execute()
            matters = r.data or []
        except:
            pass

    system = """You are a legal precedent search AI. Find relevant precedents from the database and general knowledge.

Format:
MOST RELEVANT PRECEDENTS:

1. [Precedent name / case / document]
   Relevance: [why relevant]
   Source: [from database or general knowledge]
   Application: [how to apply to current query]

GENERAL KNOWLEDGE PRECEDENTS:
[Additional relevant cases or documents from legal knowledge]

SEARCH SUMMARY:
[Overall assessment of what was found]"""

    matters_text = "\n".join([
        f"REF: {m.get('matter_ref','')} | CLIENT: {m.get('client','')} | "
        f"DESC: {m.get('description','')[:150]}"
        for m in matters[:50]
    ]) if matters else "No matters in database."

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content":
             f"Search query: {query}\n\nFirm matter database:\n{matters_text}"}
        ], max_tokens=1500)
        return ok({"result": result, "matters_searched": len(matters)})
    except Exception as e:
        return err("Precedent search failed.")

# ══════════════════════════════════════════════════════════════
# NEW: MATTER WORKSPACES
# ══════════════════════════════════════════════════════════════
@app.post("/api/v1/workspaces")
@limiter.limit("30/minute")
async def create_workspace(request: Request, data: WorkspaceCreate):
    if not supabase:
        return err("Database not connected.")
    try:
        record = {
            "firm_id": data.firm_id,
            "client_name": data.client_name,
            "matter_ref": data.matter_ref,
            "matter_type": data.matter_type,
            "jurisdiction": data.jurisdiction,
            "lead_partner": data.lead_partner,
            "status": "OPEN",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        r = supabase.table("matter_workspaces").insert(record).execute()
        return ok(r.data[0] if r.data else record)
    except Exception as e:
        return err(f"Failed to create workspace: {str(e)}")

@app.get("/api/v1/workspaces")
async def list_workspaces(firm_id: str = "default"):
    if not supabase:
        return ok([])
    try:
        r = supabase.table("matter_workspaces").select("*")\
            .eq("firm_id", firm_id)\
            .is_("deleted_at", "null")\
            .order("updated_at", desc=True).execute()
        return ok(r.data or [])
    except Exception as e:
        return err("Failed to fetch workspaces.")

@app.get("/api/v1/workspaces/{workspace_id}")
async def get_workspace(workspace_id: str):
    if not supabase:
        return err("Database not connected.")
    try:
        ws = supabase.table("matter_workspaces").select("*").eq("id", workspace_id).execute()
        items = supabase.table("workspace_items").select("*").eq("workspace_id", workspace_id)\
            .order("created_at", desc=True).execute()
        comments = supabase.table("workspace_comments").select("*").eq("workspace_id", workspace_id)\
            .order("created_at", desc=True).execute()
        return ok({
            "workspace": ws.data[0] if ws.data else {},
            "items": items.data or [],
            "comments": comments.data or []
        })
    except Exception as e:
        return err("Failed to fetch workspace.")

@app.post("/api/v1/workspaces/{workspace_id}/items")
async def save_workspace_item(workspace_id: str, data: WorkspaceItemCreate):
    if not supabase:
        return err("Database not connected.")
    try:
        r = supabase.table("workspace_items").insert({
            "workspace_id": workspace_id,
            "firm_id": data.firm_id,
            "item_type": data.item_type,
            "title": data.title,
            "content": data.content,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        supabase.table("matter_workspaces")\
            .update({"updated_at": datetime.utcnow().isoformat()})\
            .eq("id", workspace_id).execute()
        return ok(r.data[0] if r.data else {})
    except Exception as e:
        return err("Failed to save item.")

@app.delete("/api/v1/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str):
    if not supabase:
        return err("Database not connected.")
    supabase.table("matter_workspaces")\
        .update({"deleted_at": datetime.utcnow().isoformat()})\
        .eq("id", workspace_id).execute()
    return ok({"deleted": True})

# ══════════════════════════════════════════════════════════════
# NEW: HISTORY & BOOKMARKS
# ══════════════════════════════════════════════════════════════
@app.get("/api/v1/history")
async def get_history(firm_id: str = "default", limit: int = 50):
    if not supabase:
        return ok([])
    try:
        r = supabase.table("interaction_history").select("*")\
            .eq("firm_id", firm_id)\
            .order("created_at", desc=True)\
            .limit(min(limit, 200)).execute()
        return ok(r.data or [])
    except Exception as e:
        return err("Failed to fetch history.")

@app.get("/api/v1/bookmarks")
async def get_bookmarks(firm_id: str = "default"):
    if not supabase:
        return ok([])
    try:
        r = supabase.table("interaction_history").select("*")\
            .eq("firm_id", firm_id)\
            .eq("bookmarked", True)\
            .order("created_at", desc=True).execute()
        return ok(r.data or [])
    except Exception as e:
        return err("Failed to fetch bookmarks.")

@app.patch("/api/v1/history/{item_id}/bookmark")
async def toggle_bookmark(item_id: str, bookmarked: bool = True):
    if not supabase:
        return err("Database not connected.")
    supabase.table("interaction_history")\
        .update({"bookmarked": bookmarked})\
        .eq("id", item_id).execute()
    return ok({"bookmarked": bookmarked})

# ══════════════════════════════════════════════════════════════
# NEW: COLLABORATION — WORKSPACE COMMENTS
# ══════════════════════════════════════════════════════════════
@app.post("/api/v1/workspaces/{workspace_id}/comments")
async def post_comment(workspace_id: str, data: CommentCreate):
    if not supabase:
        return err("Database not connected.")
    try:
        r = supabase.table("workspace_comments").insert({
            "workspace_id": workspace_id,
            "firm_id": data.firm_id,
            "author": data.author,
            "comment": data.comment,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        return ok(r.data[0] if r.data else {})
    except Exception as e:
        return err("Failed to post comment.")

@app.get("/api/v1/workspaces/{workspace_id}/comments")
async def get_comments(workspace_id: str):
    if not supabase:
        return ok([])
    r = supabase.table("workspace_comments").select("*")\
        .eq("workspace_id", workspace_id)\
        .order("created_at", desc=True).execute()
    return ok(r.data or [])

# ══════════════════════════════════════════════════════════════
# FIRM SETTINGS
# ══════════════════════════════════════════════════════════════
@app.get("/api/v1/firm-settings")
async def get_firm_settings(firm_id: str = "default"):
    if not supabase:
        return ok({"firm_id": firm_id, "firm_name": "My Firm",
                   "jurisdiction": "UK", "show_odv": False})
    try:
        r = supabase.table("firm_settings").select("*")\
            .eq("firm_id", firm_id).execute()
        if r.data:
            return ok(r.data[0])
        return ok({"firm_id": firm_id, "firm_name": "My Firm",
                   "jurisdiction": "UK", "show_odv": False})
    except:
        return ok({"firm_id": firm_id, "firm_name": "My Firm",
                   "jurisdiction": "UK", "show_odv": False})

@app.post("/api/v1/firm-settings")
async def save_firm_settings(request: Request):
    body = await request.json()
    if not supabase:
        return ok(body)
    try:
        firm_id = body.get("firm_id", "default")
        existing = supabase.table("firm_settings").select("id")\
            .eq("firm_id", firm_id).execute()
        if existing.data:
            supabase.table("firm_settings").update(body)\
                .eq("firm_id", firm_id).execute()
        else:
            supabase.table("firm_settings").insert(body).execute()
        return ok(body)
    except Exception as e:
        return err(str(e))
