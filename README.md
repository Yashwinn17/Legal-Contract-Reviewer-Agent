# ⚖️ Day 12: Legal Contract Reviewer Agent
**84-Day Agentic AI Engineer Blueprint** | LangGraph + RAG + Self-Correction

> *AI contract review isn't a toy anymore. Here's a production-grade architecture.*

---

## 🎯 Problem & Market Opportunity

Small businesses sign contracts daily but can't afford $400/hr legal fees for routine review. The SMB legal services market is a **$50B+ gap** where AI can deliver 80% of the value at 1% of the cost.

This agent reviews contracts end-to-end: extracts clauses → checks against a FAISS RAG knowledge base of legal standards → flags risks → suggests jurisdiction-aware revisions → self-verifies.

---

## 🏗️ Architecture

```
                    ┌─────────────────────┐
                    │   Contract Input    │
                    │  (PDF / Text / API) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  EXTRACT CLAUSES    │  ← LLM parses into typed clause objects
                    │  (LangGraph Node 1) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  RAG RISK CHECK     │  ← FAISS retrieves relevant legal standards
                    │  (LangGraph Node 2) │    per clause type + jurisdiction
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  SUGGEST REVISIONS  │  ← LLM drafts safer alternative language
                    │  (LangGraph Node 3) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   SELF-VERIFY       │  ← LLM-as-judge checks revision quality
                    │  (LangGraph Node 4) │
                    └──────┬──────┬───────┘
                   (pass)  │      │  (fail, iter < max)
                           │      └──────────────────────→ Node 3 (re-revise)
                    ┌──────▼──────────────┐
                    │  GENERATE REPORT    │  ← Executive summary + risk score
                    │  (LangGraph Node 5) │
                    └─────────────────────┘
```

### Key Design Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| **Orchestration** | LangGraph StateGraph | Explicit control flow, supports loops |
| **RAG** | FAISS + curated legal KB | No hallucination on legal standards |
| **LLM** | Ollama (llama3.1:8b / qwen2.5:7b) | $0 cost, local privacy for contracts |
| **Fallback** | Gemini 1.5 Flash | When Ollama unavailable |
| **Self-correction** | Verification loop (max 2 iter) | Quality gate before report |
| **Jurisdiction** | Soft-filter RAG retrieval | State-specific rules without separate pipelines |

---

## 🚀 Quickstart

```bash
# 1. Clone and install
git clone https://github.com/Yashwinn17/day12-legal-contract-reviewer
cd day12-legal-contract-reviewer
pip install -r requirements.txt

# 2. Pull Ollama models
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 3. Configure environment
cp .env.example .env
# Add GOOGLE_API_KEY if using Gemini fallback

# 4. Run Streamlit UI
streamlit run app.py

# 5. Or use the Python API directly
python -c "
from src.agent import review_contract
result = review_contract(
    contract_text=open('contracts_sample/sample_service_agreement.txt').read(),
    jurisdiction='california',
    filename='sample.txt'
)
print(result['final_report'])
"
```

---

## 📊 Graph Nodes Deep Dive

### Node 1: Clause Extraction
- Parses contract into typed `ExtractedClause` objects
- 10 clause types: payment, termination, liability, ip, confidentiality, dispute, renewal, force_majeure, warranty, general
- Returns structured JSON for downstream processing

### Node 2: RAG Risk Check
- FAISS similarity search over 14 curated legal standard documents
- Jurisdiction-aware soft filtering (CA, NY, TX, FL, Federal, General)
- Returns `RiskFlag` objects with confidence scores

### Node 3: Revision Suggestions
- Generates plain-English alternative clause language
- Jurisdiction-specific notes per revision
- Skips already-revised clauses in re-revision iterations

### Node 4: Self-Verification (Loop)
- LLM-as-judge reviews all flags and revisions holistically
- Outputs final risk score (0–100) and recommendation
- Loops back to Node 3 if issues remain (max 2 iterations)

### Node 5: Report Generation
- Executive summary written for non-lawyers
- Highlights top 3 critical risks
- Next steps for business owner

---

## 🌐 Jurisdiction Support

| Jurisdiction | Key Rules Covered |
|---|---|
| **California** | CUTSA, Labor Code §2870, BPC auto-renewal §17601, at-will limits |
| **New York** | UCC strictness, CPLR §7515, non-compete reasonableness |
| **Texas** | Freedom of Contract, non-compete §15.50, attorney fees §38.001 |
| **Florida** | Non-compete enforcement, FL UCC |
| **Federal** | Copyright Act §101, DTSA, UCC Article 2 |
| **General** | Industry best practices, common law standards |

---

## 🧠 Engineering Lessons (Day 13)

1. **RAG + LangGraph = Grounded Agents**: Without RAG, LLM hallucinates legal standards. With FAISS retrieval, every risk flag cites a real source.

2. **Jurisdiction as Soft Filter**: Hard filters (metadata equality) miss edge cases. Soft ranking (over-fetch + re-rank) gives better results across overlapping legal domains.

3. **Self-Correction Loops Need Max Iterations**: Always cap loops. Legal review can be circular — "fix X, which creates Y" — so 2 iterations is the sweet spot.

4. **LLM Selection per Node**: Use smaller/faster model for extraction (structured output), larger/smarter model for risk assessment and revision generation.

5. **Contract Privacy**: Never send client contracts to external APIs without consent. Ollama-first architecture keeps everything local.

---

## 📁 Project Structure

```
day12-legal-contract-reviewer/
├── src/
│   ├── models.py          # Pydantic state models
│   ├── llm_provider.py    # Ollama + Gemini factory
│   ├── knowledge_base.py  # FAISS RAG legal KB
│   ├── prompts.py         # All prompt templates
│   └── agent.py           # LangGraph graph + nodes
├── contracts_sample/
│   └── sample_service_agreement.txt
├── tests/
│   └── test_knowledge_base.py
├── app.py                 # Streamlit UI
├── requirements.txt
└── README.md
```

---

## 🔮 Next Steps (Beyond Day 13)

- [ ] **Multi-document comparison**: Compare your contract vs industry templates
- [ ] **Clause negotiation history**: Track changes across contract versions
- [ ] **Case law integration**: RAG over actual court decisions
- [ ] **API endpoint**: FastAPI wrapper for B2B integration
- [ ] **GCP Deployment**: Cloud Run + Vertex AI embeddings

---

*Part of the [84-Day Agentic AI Engineer Blueprint](https://linkedin.com/in/yashwin-vasanth) | Built by Yashwin Vasanth*
