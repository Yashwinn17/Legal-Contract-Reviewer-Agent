"""
Streamlit UI for the Legal Contract Reviewer Agent.
"""
import io
import json
import time
from pathlib import Path

import PyPDF2
import streamlit as st

OLLAMA_SMALL_MODEL = "llama3.1:8b"
OLLAMA_LARGE_MODEL = "qwen2.5:7b"
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

st.set_page_config(
    page_title="LegalAI Contract Reviewer",
    page_icon="⚖️",
    layout="wide",
)

st.markdown(
    """
<style>
    .risk-critical { background: #ff000020; border-left: 4px solid #ff0000; padding: 10px; margin: 5px 0; border-radius: 4px; }
    .risk-high { background: #ff6b0020; border-left: 4px solid #ff6b00; padding: 10px; margin: 5px 0; border-radius: 4px; }
    .risk-medium { background: #ffd70020; border-left: 4px solid #ffd700; padding: 10px; margin: 5px 0; border-radius: 4px; }
    .risk-low { background: #00ff0020; border-left: 4px solid #00aa00; padding: 10px; margin: 5px 0; border-radius: 4px; }
    .approve-badge { background: #00aa0030; color: #00aa00; padding: 6px 16px; border-radius: 20px; font-weight: bold; }
    .review-badge { background: #ffd70030; color: #aa8800; padding: 6px 16px; border-radius: 20px; font-weight: bold; }
    .reject-badge { background: #ff000030; color: #cc0000; padding: 6px 16px; border-radius: 20px; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("LegalAI Contract Reviewer")
st.caption("Day 12 | 84-Day Agentic AI Engineer Blueprint | LangGraph + RAG")
st.divider()

with st.sidebar:
    st.header("Review Settings")

    llm_provider = st.radio(
        "LLM Provider",
        options=["ollama", "gemini"],
        horizontal=True,
        help="Choose your local Ollama models or Gemini for the full review flow.",
    )

    if llm_provider == "ollama":
        st.caption(f"Models used: `{OLLAMA_SMALL_MODEL}` and `{OLLAMA_LARGE_MODEL}`")
    else:
        st.caption(f"Model used: `{GEMINI_MODEL}`")

    jurisdiction = st.selectbox(
        "Jurisdiction",
        options=["general", "california", "new_york", "texas", "florida", "federal"],
        index=0,
        help="Select jurisdiction for state-specific legal analysis.",
    )

    st.info(f"Jurisdiction-aware analysis: **{jurisdiction.replace('_', ' ').title()}**")

    st.divider()
    st.subheader("Knowledge Base")
    st.success("FAISS index: 14 legal standards")
    st.success("Clause types: 10 categories")
    st.success("Jurisdictions: 6 covered")

    st.divider()
    st.markdown("**Graph Flow:**")
    st.markdown(
        """
1. Extract Clauses
2. RAG Risk Check
3. Flag Risks
4. Suggest Revisions
5. Self-Verify (loop)
6. Generate Report
"""
    )

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Contract")

    input_method = st.radio(
        "Input method",
        ["Upload PDF", "Paste Text", "Use Sample Contract"],
        horizontal=True,
    )

    contract_text = ""
    filename = "contract.txt"

    if input_method == "Upload PDF":
        uploaded = st.file_uploader("Upload contract PDF", type=["pdf"])
        if uploaded:
            filename = uploaded.name
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded.read()))
            contract_text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            st.success(f"Loaded: {filename} ({len(reader.pages)} pages)")
            if not contract_text:
                st.warning("No text could be extracted from this PDF. Try paste text instead.")

    elif input_method == "Paste Text":
        contract_text = st.text_area(
            "Paste contract text",
            height=300,
            placeholder="Paste the full contract text here...",
        )
        filename = "pasted_contract.txt"

    else:
        contract_text = Path("contracts_sample/sample_service_agreement.txt").read_text()
        filename = "sample_service_agreement.txt"
        st.info("Sample service agreement loaded")
        with st.expander("Preview sample contract"):
            st.text(contract_text[:800] + "...")

    analyze_btn = st.button(
        "Analyze Contract",
        type="primary",
        disabled=not contract_text,
        use_container_width=True,
    )

with col2:
    st.subheader("Review Results")
    if "review_result" not in st.session_state:
        st.info("Upload a contract and click **Analyze Contract** to begin.")

if analyze_btn and contract_text:
    with st.spinner("Running LangGraph contract review pipeline..."):
        progress = st.progress(0, text="Extracting clauses...")
        time.sleep(0.5)
        try:
            from src.agent import review_contract

            result = review_contract(
                contract_text=contract_text,
                jurisdiction=jurisdiction,
                filename=filename,
                llm_provider=llm_provider,
            )
            progress.progress(100, text="Complete!")
            st.session_state["review_result"] = result
            st.session_state["review_provider"] = llm_provider
            st.session_state["review_filename"] = filename
            st.session_state.pop("review_error", None)
            st.rerun()
        except Exception as exc:
            st.session_state["review_error"] = str(exc)
            progress.empty()

if "review_result" in st.session_state:
    result = st.session_state["review_result"]
    review_provider = st.session_state.get("review_provider", llm_provider)
    export_filename = st.session_state.get("review_filename", filename)
    verification = result.get("verification_result", {})
    recommendation = verification.get("recommendation", "review")
    risk_score = verification.get("final_risk_score", 50)

    st.caption(f"Reviewed with: `{review_provider}`")

    if result.get("errors"):
        for error in result["errors"]:
            st.warning(error)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clauses Analyzed", len(result.get("extracted_clauses", [])))
    m2.metric("Risk Flags", len(result.get("risk_flags", [])))
    m3.metric("Revisions Generated", len(result.get("revision_suggestions", [])))
    m4.metric("Risk Score", f"{risk_score:.0f}/100")

    badge_map = {
        "approve": ("APPROVE", "approve-badge"),
        "review": ("NEEDS REVIEW", "review-badge"),
        "reject": ("REJECT / RENEGOTIATE", "reject-badge"),
    }
    label, css_class = badge_map.get(recommendation, ("REVIEW", "review-badge"))
    st.markdown(f'<span class="{css_class}">{label}</span>', unsafe_allow_html=True)

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["Report", "Risk Flags", "Revisions", "Clauses"])

    with tab1:
        st.markdown(result.get("final_report", "Report not generated."))

    with tab2:
        flags = result.get("risk_flags", [])
        if not flags:
            st.success("No significant risks flagged.")
        for flag in sorted(flags, key=lambda item: ["critical", "high", "medium", "low"].index(item["risk_level"])):
            css = f"risk-{flag['risk_level']}"
            st.markdown(
                f"""
<div class="{css}">
    <strong>[{flag['risk_level'].upper()}] {flag['clause_id']} - {flag['clause_type'].title()}</strong><br>
    {flag['risk_description']}<br>
    <small>Standard: {flag['legal_standard_violated']} | Source: {flag['rag_source']} | Confidence: {flag['confidence_score']:.0%}</small>
</div>
""",
                unsafe_allow_html=True,
            )

    with tab3:
        revisions = result.get("revision_suggestions", [])
        if not revisions:
            st.info("No revisions generated.")
        for revision in revisions:
            with st.expander(f"{revision['clause_id']} - Suggested Revision"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Original:**")
                    st.text_area(
                        "Original Clause",
                        revision["original_text"],
                        key=f"orig_{revision['clause_id']}",
                        height=100,
                    )
                with col_b:
                    st.markdown("**Suggested:**")
                    st.text_area(
                        "Suggested Clause",
                        revision["suggested_text"],
                        key=f"sugg_{revision['clause_id']}",
                        height=100,
                    )
                st.caption(f"Reasoning: {revision['reasoning']}")
                if revision.get("jurisdiction_notes"):
                    st.caption(f"{jurisdiction.title()} note: {revision['jurisdiction_notes']}")

    with tab4:
        for clause in result.get("extracted_clauses", []):
            with st.expander(f"{clause['clause_id']} - {clause['clause_type'].title()}"):
                st.text(clause["clause_text"])

    st.divider()
    st.download_button(
        "Export Review as JSON",
        data=json.dumps(result, indent=2),
        file_name=f"review_{export_filename}.json",
        mime="application/json",
    )

if "review_error" in st.session_state:
    st.error(f"Review failed: {st.session_state['review_error']}")
