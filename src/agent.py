"""
Legal Contract Reviewer Agent.

Workflow:
Extract -> RAG Check -> Suggest Revisions -> Self-Verify -> Generate Report
"""
import json
import os
import re
from typing import List, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from src.knowledge_base import LegalKnowledgeBase
from src.llm_provider import get_llm
from src.models import ExtractedClause, RevisionSuggestion, RiskFlag, RiskLevel, VerificationResult
from src.prompts import (
    CLAUSE_EXTRACTION_PROMPT,
    REPORT_GENERATION_PROMPT,
    REVISION_PROMPT,
    RISK_ASSESSMENT_PROMPT,
    VERIFICATION_PROMPT,
)


def _response_to_text(response) -> str:
    """Normalize provider responses into plain text."""
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
                else:
                    parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def _invoke_llm(prompt: str, provider: str, temperature: float = 0.0, prefer_large: bool = False):
    """Invoke the requested provider and fall back to Gemini on Ollama memory errors."""
    try:
        llm = get_llm(
            temperature=temperature,
            prefer_large=prefer_large,
            provider=provider,
        )
        return llm.invoke([HumanMessage(content=prompt)])
    except Exception as exc:
        error_text = str(exc).lower()
        can_fallback = provider == "ollama" and os.getenv("GOOGLE_API_KEY")

        if can_fallback and "requires more system memory" in error_text:
            print("[LLM] Ollama memory limit hit, retrying with Gemini...")
            fallback_llm = get_llm(
                temperature=temperature,
                prefer_large=False,
                provider="gemini",
            )
            return fallback_llm.invoke([HumanMessage(content=prompt)])

        raise


class GraphState(TypedDict):
    contract_text: str
    jurisdiction: str
    contract_filename: str
    llm_provider: Literal["ollama", "gemini"]
    extracted_clauses: List[dict]
    risk_flags: List[dict]
    revision_suggestions: List[dict]
    verification_result: dict
    revision_iteration: int
    max_revision_iterations: int
    errors: List[str]
    final_report: str


def extract_clauses_node(state: GraphState) -> GraphState:
    print("[Node 1] Extracting clauses...")
    prompt = CLAUSE_EXTRACTION_PROMPT.format(contract_text=state["contract_text"][:4000])

    try:
        response = _invoke_llm(
            prompt=prompt,
            provider=state["llm_provider"],
            temperature=0.0,
        )
        raw = re.sub(r"```json\s*|\s*```", "", _response_to_text(response)).strip()
        clauses_data = json.loads(raw)
        clauses = [ExtractedClause(**clause).model_dump() for clause in clauses_data]
        print(f"[Node 1] Extracted {len(clauses)} clauses")
        return {**state, "extracted_clauses": clauses}
    except Exception as exc:
        error = f"Clause extraction failed: {exc}"
        print(f"[Node 1] ERROR: {error}")
        return {**state, "errors": state["errors"] + [error]}


def check_clauses_rag_node(state: GraphState) -> GraphState:
    print("[Node 2] RAG risk assessment...")
    kb = LegalKnowledgeBase(embedding_provider=state["llm_provider"])
    risk_flags = []

    for clause in state["extracted_clauses"]:
        relevant_docs = kb.retrieve(
            query=clause["clause_text"],
            clause_type=clause["clause_type"],
            jurisdiction=state["jurisdiction"],
            k=3,
        )
        rag_context = "\n\n".join(
            f"[{doc.metadata.get('source', 'Legal Standard')}]\n{doc.page_content}"
            for doc in relevant_docs
        )
        prompt = RISK_ASSESSMENT_PROMPT.format(
            clause_type=clause["clause_type"],
            clause_text=clause["clause_text"],
            rag_context=rag_context,
            jurisdiction=state["jurisdiction"],
        )

        try:
            response = _invoke_llm(
                prompt=prompt,
                provider=state["llm_provider"],
                temperature=0.0,
                prefer_large=True,
            )
            raw = re.sub(r"```json\s*|\s*```", "", _response_to_text(response)).strip()
            risk_data = json.loads(raw)

            flag = RiskFlag(
                clause_id=clause["clause_id"],
                clause_type=clause["clause_type"],
                risk_level=risk_data["risk_level"],
                risk_description=risk_data["risk_description"],
                legal_standard_violated=risk_data["legal_standard_violated"],
                rag_source=(
                    relevant_docs[0].metadata.get("source", "Legal KB")
                    if relevant_docs
                    else "General"
                ),
                confidence_score=risk_data.get("confidence_score", 0.8),
            )

            if flag.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL):
                risk_flags.append(flag.model_dump())
                print(f"[Node 2] {clause['clause_id']} ({clause['clause_type']}): {flag.risk_level.value}")
        except Exception as exc:
            print(f"[Node 2] Skipping {clause['clause_id']}: {exc}")

    print(f"[Node 2] Found {len(risk_flags)} risk flags")
    return {**state, "risk_flags": risk_flags}


def suggest_revisions_node(state: GraphState) -> GraphState:
    iteration = state["revision_iteration"] + 1
    print(f"[Node 3] Generating revisions (iteration {iteration})...")
    clause_lookup = {clause["clause_id"]: clause["clause_text"] for clause in state["extracted_clauses"]}
    target_flags = [
        flag
        for flag in state["risk_flags"]
        if flag["risk_level"] in ("high", "critical") or iteration > 1
    ]
    revisions = list(state.get("revision_suggestions", []))

    for flag in target_flags:
        if any(revision["clause_id"] == flag["clause_id"] for revision in revisions):
            continue

        prompt = REVISION_PROMPT.format(
            original_text=clause_lookup.get(flag["clause_id"], "Clause text not found"),
            risk_description=flag["risk_description"],
            legal_standard_violated=flag["legal_standard_violated"],
            jurisdiction=state["jurisdiction"],
        )

        try:
            response = _invoke_llm(
                prompt=prompt,
                provider=state["llm_provider"],
                temperature=0.1,
                prefer_large=True,
            )
            raw = re.sub(r"```json\s*|\s*```", "", _response_to_text(response)).strip()
            revision_data = json.loads(raw)
            revision = RevisionSuggestion(
                clause_id=flag["clause_id"],
                original_text=clause_lookup.get(flag["clause_id"], "Clause text not found"),
                suggested_text=revision_data["suggested_text"],
                reasoning=revision_data["reasoning"],
                jurisdiction_notes=revision_data.get("jurisdiction_notes"),
            )
            revisions.append(revision.model_dump())
            print(f"[Node 3] Revised {flag['clause_id']}")
        except Exception as exc:
            print(f"[Node 3] Revision failed for {flag['clause_id']}: {exc}")

    return {**state, "revision_suggestions": revisions, "revision_iteration": iteration}


def verify_revisions_node(state: GraphState) -> GraphState:
    print("[Node 4] Self-verification...")
    high_risk_count = sum(1 for flag in state["risk_flags"] if flag["risk_level"] in ("high", "critical"))
    risk_flags_summary = "\n".join(
        f"- [{flag['risk_level'].upper()}] {flag['clause_id']} ({flag['clause_type']}): {flag['risk_description'][:100]}"
        for flag in state["risk_flags"]
    )
    revisions_summary = "\n".join(
        f"- {revision['clause_id']}: {revision['reasoning'][:100]}"
        for revision in state["revision_suggestions"]
    )
    prompt = VERIFICATION_PROMPT.format(
        clause_count=len(state["extracted_clauses"]),
        risk_count=len(state["risk_flags"]),
        high_risk_count=high_risk_count,
        revision_count=len(state["revision_suggestions"]),
        risk_flags_summary=risk_flags_summary or "No risks flagged",
        revisions_summary=revisions_summary or "No revisions generated",
    )

    try:
        response = _invoke_llm(
            prompt=prompt,
            provider=state["llm_provider"],
            temperature=0.0,
        )
        raw = re.sub(r"```json\s*|\s*```", "", _response_to_text(response)).strip()
        verification_data = json.loads(raw)
        result = VerificationResult(**verification_data)
        print(f"[Node 4] Verification: {result.recommendation} (score: {result.final_risk_score})")
        return {**state, "verification_result": result.model_dump()}
    except Exception as exc:
        print(f"[Node 4] Verification error: {exc}")
        fallback = VerificationResult(
            verified=True,
            issues_found=[],
            final_risk_score=50.0,
            recommendation="review",
        )
        return {**state, "verification_result": fallback.model_dump()}


def generate_report_node(state: GraphState) -> GraphState:
    print("[Node 5] Generating final report...")
    verification = state.get("verification_result", {})
    risk_flags_text = "\n".join(
        f"- [{flag['risk_level'].upper()}] {flag['clause_type'].title()}: {flag['risk_description']}"
        for flag in state["risk_flags"]
    ) or "No significant risks identified."
    revisions_text = "\n".join(
        f"- {revision['clause_id']} ({state['jurisdiction']}): {revision['reasoning']}"
        for revision in state["revision_suggestions"]
    ) or "No revisions required."
    prompt = REPORT_GENERATION_PROMPT.format(
        filename=state["contract_filename"],
        jurisdiction=state["jurisdiction"],
        clause_count=len(state["extracted_clauses"]),
        recommendation=verification.get("recommendation", "review").upper(),
        risk_score=verification.get("final_risk_score", 50),
        risk_flags=risk_flags_text,
        revisions=revisions_text,
    )
    response = _invoke_llm(
        prompt=prompt,
        provider=state["llm_provider"],
        temperature=0.2,
    )
    print("[Node 5] Report generated")
    return {**state, "final_report": _response_to_text(response)}


def should_re_revise(state: GraphState) -> str:
    verification = state.get("verification_result", {})
    issues = verification.get("issues_found", [])
    iteration = state.get("revision_iteration", 0)
    max_iterations = state.get("max_revision_iterations", 2)

    if issues and iteration < max_iterations:
        print(f"[Router] Issues found, re-revising (iteration {iteration}/{max_iterations})")
        return "suggest_revisions"

    print("[Router] Verification passed, generating report")
    return "generate_report"


def build_contract_review_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("extract_clauses", extract_clauses_node)
    graph.add_node("check_clauses_rag", check_clauses_rag_node)
    graph.add_node("suggest_revisions", suggest_revisions_node)
    graph.add_node("verify_revisions", verify_revisions_node)
    graph.add_node("generate_report", generate_report_node)
    graph.set_entry_point("extract_clauses")
    graph.add_edge("extract_clauses", "check_clauses_rag")
    graph.add_edge("check_clauses_rag", "suggest_revisions")
    graph.add_edge("suggest_revisions", "verify_revisions")
    graph.add_conditional_edges(
        "verify_revisions",
        should_re_revise,
        {
            "suggest_revisions": "suggest_revisions",
            "generate_report": "generate_report",
        },
    )
    graph.add_edge("generate_report", END)
    return graph.compile()


def review_contract(
    contract_text: str,
    jurisdiction: str = "general",
    filename: str = "contract.pdf",
    llm_provider: Literal["ollama", "gemini"] = "ollama",
) -> GraphState:
    graph = build_contract_review_graph()
    initial_state: GraphState = {
        "contract_text": contract_text,
        "jurisdiction": jurisdiction,
        "contract_filename": filename,
        "llm_provider": llm_provider,
        "extracted_clauses": [],
        "risk_flags": [],
        "revision_suggestions": [],
        "verification_result": {},
        "revision_iteration": 0,
        "max_revision_iterations": 2,
        "errors": [],
        "final_report": "",
    }
    return graph.invoke(initial_state)
