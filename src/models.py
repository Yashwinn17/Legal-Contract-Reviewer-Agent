"""
Pydantic models for the Legal Contract Reviewer Agent.
"""
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Jurisdiction(str, Enum):
    FEDERAL = "federal"
    CALIFORNIA = "california"
    NEW_YORK = "new_york"
    TEXAS = "texas"
    FLORIDA = "florida"
    GENERAL = "general"


class ExtractedClause(BaseModel):
    clause_id: str
    clause_type: str
    clause_text: str
    page_reference: Optional[str] = None


class RiskFlag(BaseModel):
    clause_id: str
    clause_type: str
    risk_level: RiskLevel
    risk_description: str
    legal_standard_violated: str
    rag_source: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class RevisionSuggestion(BaseModel):
    clause_id: str
    original_text: str
    suggested_text: str
    reasoning: str
    jurisdiction_notes: Optional[str] = None


class VerificationResult(BaseModel):
    verified: bool
    issues_found: List[str]
    final_risk_score: float
    recommendation: Literal["approve", "review", "reject"]


class ContractReviewState(BaseModel):
    """LangGraph state passed between all graph nodes."""

    contract_text: str
    jurisdiction: Jurisdiction = Jurisdiction.GENERAL
    contract_filename: str = "contract.pdf"
    llm_provider: Literal["ollama", "gemini"] = "ollama"
    extracted_clauses: List[ExtractedClause] = Field(default_factory=list)
    risk_flags: List[RiskFlag] = Field(default_factory=list)
    revision_suggestions: List[RevisionSuggestion] = Field(default_factory=list)
    verification_result: Optional[VerificationResult] = None
    revision_iteration: int = 0
    max_revision_iterations: int = 2
    errors: List[str] = Field(default_factory=list)
    final_report: str = ""
