#!/usr/bin/env python3
"""
backend/main.py: FastAPI REST server for the PyRete ReAct Hardware Pricing Coordinator.
Serves on standard port 8000 with CORS enabled for Next.js frontend on port 3000.
Supports session cookies and session tracking for multi-user working memory isolation.
"""

import os
import uuid
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from coordinator import PyReteCoordinatorEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

app = FastAPI(
    title="PyRete Hardware Pricing Agent API",
    description="Goal-driven forward chaining rule coordinator for hardware pricing & international conversions",
    version="1.1.0",
)

# Enable CORS for Next.js dev server on port 3000 with credentials support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Predefined hardware products
HARDWARE_PRODUCTS = [
    {"id": "macbook_pro_14", "name": "Apple MacBook Pro 14\"", "category": "Laptop", "brand": "Apple"},
    {"id": "dell_xps_15", "name": "Dell XPS 15", "category": "Laptop", "brand": "Dell"},
    {"id": "lenovo_thinkpad_x1", "name": "Lenovo ThinkPad X1 Carbon", "category": "Laptop", "brand": "Lenovo"},
    {"id": "asus_rog_g14", "name": "ASUS ROG Zephyrus G14", "category": "Gaming Laptop", "brand": "ASUS"},
    {"id": "ipad_pro_m4", "name": "Apple iPad Pro M4", "category": "Tablet", "brand": "Apple"},
    {"id": "galaxy_tab_s9", "name": "Samsung Galaxy Tab S9", "category": "Tablet", "brand": "Samsung"},
    {"id": "hp_spectre_x360", "name": "HP Spectre x360", "category": "2-in-1 Laptop", "brand": "HP"},
]

# Supported international currencies with benchmark conversion rates
CURRENCIES = [
    {"code": "EUR", "name": "Euro", "symbol": "€", "approx_rate": 0.85},
    {"code": "GBP", "name": "British Pound", "symbol": "£", "approx_rate": 0.77},
    {"code": "JPY", "name": "Japanese Yen", "symbol": "¥", "approx_rate": 150.0},
    {"code": "CAD", "name": "Canadian Dollar", "symbol": "C$", "approx_rate": 1.36},
    {"code": "AUD", "name": "Australian Dollar", "symbol": "A$", "approx_rate": 1.52},
    {"code": "CHF", "name": "Swiss Franc", "symbol": "Fr", "approx_rate": 0.88},
    {"code": "INR", "name": "Indian Rupee", "symbol": "₹", "approx_rate": 83.5},
]

# Shared coordinator instance with compiled production rules
coordinator_engine = PyReteCoordinatorEngine(max_iterations=20)


# =============================================================================
# Request & Response Schemas
# =============================================================================

class PricingRequest(BaseModel):
    product: str = Field(..., example="Apple MacBook Pro 14\"")
    currency_code: str = Field(..., example="EUR")
    currency_name: Optional[str] = Field("Euro", example="Euro")
    exchange_rate: float = Field(0.85, example=0.85)
    custom_query: Optional[str] = None
    session_id: Optional[str] = Field(None, example="sess-123456")


class AuditItem(BaseModel):
    step: Optional[int] = None
    kind: str
    content: str


class PricingResponse(BaseModel):
    success: bool
    session_id: str
    product: str
    currency: str
    exchange_rate: float
    query: str
    final_answer: str
    status: str
    cached: bool = False
    audit_facts: List[AuditItem] = []


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "engine": "PyRete ReAct",
        "llm_backend": "Google Gemini",
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    }


@app.get("/api/options")
def get_options():
    """Returns the guard-railed hardware and currency options for the frontend."""
    return {
        "products": HARDWARE_PRODUCTS,
        "currencies": CURRENCIES,
    }


@app.post("/api/pricing", response_model=PricingResponse)
def compute_pricing(
    req: PricingRequest,
    request: Request,
    response: Response,
):
    """
    Synthesizes the guard-railed query and executes within the session's Working Memory scope.
    Checks cached WM first, then Gemini accessibility, and clears WM upon Finished resolution.
    """
    # Resolve or create session identifier
    session_id = (
        req.session_id
        or request.cookies.get("session_id")
        or f"sess-{uuid.uuid4().hex[:12]}"
    )

    # Set or refresh session cookie
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=False,
        samesite="lax",
    )

    if req.custom_query and req.custom_query.strip():
        query = req.custom_query.strip()
    else:
        query = (
            f"What is the total retail purchase price (MSRP) of a new entry-level {req.product} in USD? "
            f"Do not use monthly financing. "
            f"How much would it cost in {req.currency_name or req.currency_code} ({req.currency_code}) "
            f"if the exchange rate is {req.exchange_rate} {req.currency_code} for 1 USD?"
        )

    logger.info(f"[Session {session_id}] Query for '{req.product}' -> {req.currency_code}: {query}")

    try:
        result = coordinator_engine.run_detailed(query, session_id=session_id)
        return PricingResponse(
            success=result["success"],
            session_id=result.get("session_id", session_id),
            product=req.product,
            currency=req.currency_code,
            exchange_rate=req.exchange_rate,
            query=query,
            final_answer=result["final_answer"],
            status=result["status"],
            cached=result.get("cached", False),
            audit_facts=[AuditItem(**item) for item in result.get("audit", [])],
        )
    except Exception as e:
        logger.error(f"[Session {session_id}] Coordinator execution error: {e}", exc_info=True)
        return PricingResponse(
            success=False,
            session_id=session_id,
            product=req.product,
            currency=req.currency_code,
            exchange_rate=req.exchange_rate,
            query=query,
            final_answer=f"Sorry, I couldn't complete the request due to an unexpected execution issue: {e}",
            status="error",
            cached=False,
            audit_facts=[],
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
