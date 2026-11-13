
import os
import traceback
import uuid
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.intake_engine import (
    run_intake_turn,
    init_database,
    list_intake_records,
    get_intake_record,
    get_available_slots,
    create_booking,
    get_db,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
init_database()

app = FastAPI(
    title="IntakeIQ API",
    description="Agentic Universal Client Intake Agent",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10000)


class IntakeChatRequest(BaseModel):
    industry: str
    messages: list[ChatMessage]
    current_fields: dict[str, Any] = Field(default_factory=dict)
    intake_id: str | None = None


class BookingRequest(BaseModel):
    intake_id: str
    slot: str
    confirmed: bool = False


@app.get("/")
def home():
    return {
        "message": "Welcome to IntakeIQ API",
        "status": "running",
        "version": "2.0.0",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "ai_configured": bool(os.getenv("GEMINI_API_KEY")),
    }


@app.get("/api/dashboard/stats")
def dashboard_stats():
    records = list_intake_records()
    return {
        "total_clients": len(records),
        "active_intakes": sum(
            r["status"] == "In progress" for r in records
        ),
        "completed_intakes": sum(
            r["validation"]["is_complete"] for r in records
        ),
        "pending_reviews": sum(
            r["status"] == "Ready for review" for r in records
        ),
    }


@app.post("/api/agent/chat")
def agent_chat(request: IntakeChatRequest):
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="At least one message is required.",
        )

    try:
        return run_intake_turn(
            industry=request.industry,
            messages=[m.model_dump() for m in request.messages],
            current_fields=request.current_fields,
            intake_id=request.intake_id or str(uuid.uuid4()),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        print("\n========== INTAKEIQ ERROR ==========")
        traceback.print_exc()
        print("===================================\n")
        raise HTTPException(status_code=503, detail=str(error))


@app.get("/api/intakes")
def get_intakes():
    return {"intakes": list_intake_records()}


@app.get("/api/intakes/{intake_id}")
def get_intake(intake_id: str):
    record = get_intake_record(intake_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail="Intake not found.",
        )
    return record


@app.get("/api/consultations/slots")
def consultation_slots():
    return {"slots": get_available_slots()}


@app.post("/api/consultations/book")
def book_consultation(request: BookingRequest):
    if not request.confirmed:
        raise HTTPException(
            status_code=400,
            detail="Client confirmation is required before booking.",
        )

    try:
        return create_booking(request.intake_id, request.slot)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/api/consultations/bookings")
def get_bookings():
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM bookings ORDER BY created_at DESC"
        ).fetchall()

    return {"bookings": [dict(row) for row in rows]}
