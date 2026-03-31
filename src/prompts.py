"""
Prompt templates for Legal Contract Reviewer Agent — Day 12
"""

CLAUSE_EXTRACTION_PROMPT = """You are a legal document analyst. Extract all distinct clauses from this contract.

For each clause, identify:
1. clause_id: sequential ID (C001, C002, ...)
2. clause_type: one of [payment, termination, liability, ip, confidentiality, dispute, renewal, force_majeure, warranty, general]
3. clause_text: the exact clause text (condensed if very long, max 500 chars)

Contract Text:
{contract_text}

Return ONLY a valid JSON array. No preamble, no explanation.
Format:
[
  {{"clause_id": "C001", "clause_type": "payment", "clause_text": "..."}},
  ...
]"""


RISK_ASSESSMENT_PROMPT = """You are a legal risk analyst reviewing a contract clause against legal standards.

CONTRACT CLAUSE:
Type: {clause_type}
Text: {clause_text}

RELEVANT LEGAL STANDARDS (from knowledge base):
{rag_context}

JURISDICTION: {jurisdiction}

Assess the risk of this clause. Return ONLY valid JSON:
{{
  "risk_level": "low|medium|high|critical",
  "risk_description": "specific description of the risk",
  "legal_standard_violated": "which standard or law is potentially violated",
  "confidence_score": 0.0-1.0
}}

Risk levels:
- low: minor issue, standard practice deviation
- medium: notable concern, should be negotiated
- high: significant legal exposure, strongly recommend revision
- critical: potential unenforceability or serious legal liability"""


REVISION_PROMPT = """You are a contract attorney drafting revised contract language.

ORIGINAL CLAUSE:
{original_text}

RISK IDENTIFIED: {risk_description}
LEGAL STANDARD: {legal_standard_violated}
JURISDICTION: {jurisdiction}

Draft a revised version that:
1. Addresses the identified risk
2. Is fair to both parties
3. Uses clear, plain English
4. Complies with {jurisdiction} law

Return ONLY valid JSON:
{{
  "suggested_text": "the revised clause text",
  "reasoning": "why this revision addresses the risk",
  "jurisdiction_notes": "any jurisdiction-specific considerations"
}}"""


VERIFICATION_PROMPT = """You are a senior legal reviewer performing final quality check.

CONTRACT REVIEW SUMMARY:
Original clauses analyzed: {clause_count}
Risk flags identified: {risk_count}
High/Critical risks: {high_risk_count}
Revision suggestions provided: {revision_count}

RISK FLAGS:
{risk_flags_summary}

REVISION QUALITY CHECK:
{revisions_summary}

Perform final verification. Return ONLY valid JSON:
{{
  "verified": true|false,
  "issues_found": ["list of remaining issues after revisions"],
  "final_risk_score": 0-100,
  "recommendation": "approve|review|reject"
}}

Score guide: 0-30=low risk (approve), 31-60=moderate (review), 61-100=high risk (reject)"""


REPORT_GENERATION_PROMPT = """Generate a professional contract review report.

CONTRACT: {filename}
JURISDICTION: {jurisdiction}
CLAUSES ANALYZED: {clause_count}
RECOMMENDATION: {recommendation}
FINAL RISK SCORE: {risk_score}/100

RISK FLAGS:
{risk_flags}

REVISION SUGGESTIONS:
{revisions}

Write a clear, professional executive summary report (300-400 words) that:
1. States the overall recommendation prominently
2. Lists the top 3 most critical risks
3. Summarizes key revisions needed
4. Provides next steps for the business owner
5. Includes a jurisdiction-specific note

Use plain English. This is for a small business owner, not a lawyer."""
