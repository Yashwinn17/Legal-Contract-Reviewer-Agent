"""
RAG Knowledge Base backed by a FAISS vector store over legal standards.
"""
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.llm_provider import get_embedding_model

LEGAL_STANDARDS = [
    {
        "content": """PAYMENT TERMS STANDARD: Net-30 is the industry standard for B2B contracts.
        Late payment penalties exceeding 1.5% per month (18% APR) may be unenforceable in many jurisdictions.
        Contracts must specify: payment method, currency, invoice requirements, and dispute resolution for invoices.
        California law (Civil Code 3302) limits certain penalty clauses. Texas Business & Commerce Code §2.718
        restricts liquidated damages to reasonable estimates of actual harm.""",
        "metadata": {"type": "payment", "jurisdiction": "general", "source": "UCC + State Law"},
    },
    {
        "content": """PAYMENT RED FLAGS: Watch for: (1) Unilateral right to withhold payment without notice,
        (2) No dispute resolution process for contested invoices, (3) Payment terms exceeding Net-60 without
        interest provisions, (4) Automatic renewal of payment obligations without notice requirements,
        (5) No cap on late fees or compound interest clauses.""",
        "metadata": {"type": "payment", "jurisdiction": "general", "source": "Contract Risk Standards"},
    },
    {
        "content": """TERMINATION CLAUSE STANDARD: Contracts should specify: (1) Notice period - minimum 30 days
        for termination for convenience, (2) Cure period - minimum 15-30 days for termination for cause,
        (3) Obligations upon termination (data return, IP transfer, final payment),
        (4) Survival clauses identifying which provisions survive termination.
        Termination for convenience without any notice is considered unconscionable in most jurisdictions.""",
        "metadata": {"type": "termination", "jurisdiction": "general", "source": "Contract Best Practices"},
    },
    {
        "content": """CALIFORNIA TERMINATION RULES: California Labor Code and Business & Professions Code impose
        additional requirements. At-will termination clauses in service contracts must not waive statutory rights.
        California Courts (Guz v. Bechtel) have established implied covenant of good faith even in at-will contracts.
        Termination clauses that violate public policy are void under CA Civil Code §1668.""",
        "metadata": {"type": "termination", "jurisdiction": "california", "source": "CA Law"},
    },
    {
        "content": """LIABILITY LIMITATION STANDARD: Mutual limitation of liability clauses are standard.
        One-sided liability caps (only limiting one party) are red flags. Common cap structures:
        (1) Cap at total contract value, (2) Cap at 12 months of fees, (3) Insurance policy limits.
        Exclusions for gross negligence, willful misconduct, IP infringement, and confidentiality breaches
        are standard and should NOT be removed.""",
        "metadata": {"type": "liability", "jurisdiction": "general", "source": "Commercial Contract Standards"},
    },
    {
        "content": """INDEMNIFICATION RED FLAGS: (1) Unlimited, uncapped indemnification obligations - unacceptable,
        (2) Indemnification for third-party IP infringement without knowledge qualifier,
        (3) Defense obligations without right to control defense, (4) Indemnification triggered by negligence
        of the indemnified party, (5) No mutual indemnification (only one party indemnifies).
        New York courts (Hooper Associates v. AGS Computers) require express intent for broad indemnification.""",
        "metadata": {"type": "liability", "jurisdiction": "general", "source": "NY Law + Risk Standards"},
    },
    {
        "content": """IP OWNERSHIP STANDARD: Work-for-hire provisions must comply with Copyright Act §101.
        Categories of work-for-hire are strictly defined. For independent contractors, IP assignment must be
        explicit and in writing. California Labor Code §2870 protects employees' pre-existing IP.
        Contracts should clearly define: Background IP (pre-existing), Foreground IP (created during contract),
        and Joint IP ownership provisions.""",
        "metadata": {"type": "ip", "jurisdiction": "general", "source": "Copyright Act + CA Labor Code"},
    },
    {
        "content": """IP RED FLAGS: (1) Broad assignment of all IP including pre-existing work,
        (2) No license back to contractor for tools/frameworks used, (3) Perpetual, irrevocable license
        without compensation, (4) Assignment of moral rights (unenforceable in US),
        (5) No warranty that assigned IP doesn't infringe third-party rights.""",
        "metadata": {"type": "ip", "jurisdiction": "general", "source": "IP Risk Standards"},
    },
    {
        "content": """CONFIDENTIALITY STANDARD: Reasonable NDA provisions: (1) Duration 2-5 years
        (perpetual NDAs for trade secrets are acceptable), (2) Standard exclusions: publicly available info,
        independently developed, received from third party, required by law, (3) Limited disclosure to
        employees/contractors with need-to-know, (4) Return/destroy obligations upon termination.
        California Trade Secrets Act (CUTSA) preempts most common law claims.""",
        "metadata": {"type": "confidentiality", "jurisdiction": "california", "source": "CUTSA + DTSA"},
    },
    {
        "content": """CONFIDENTIALITY RED FLAGS: (1) No standard exclusions from confidential information,
        (2) Prohibition on disclosures required by law or court order, (3) Non-compete embedded in NDA
        (unenforceable in California), (4) Disparagement clauses beyond confidentiality scope,
        (5) Unilateral NDA where only one party has obligations.""",
        "metadata": {"type": "confidentiality", "jurisdiction": "general", "source": "NDA Risk Standards"},
    },
    {
        "content": """DISPUTE RESOLUTION STANDARD: Best practices include: (1) Mandatory negotiation/mediation
        before arbitration/litigation, (2) Specified arbitration body (AAA, JAMS) with clear rules,
        (3) Venue/jurisdiction clause specifying governing state and courts,
        (4) Jury trial waiver must be conspicuous to be enforceable,
        (5) Class action waiver enforceability varies by state (banned in employment in CA).""",
        "metadata": {"type": "dispute", "jurisdiction": "general", "source": "Dispute Resolution Standards"},
    },
    {
        "content": """TEXAS CONTRACT LAW SPECIFICS: Texas follows Freedom of Contract broadly.
        Texas Business & Commerce Code governs commercial contracts. Key rules:
        (1) Non-competes enforceable if reasonable in scope, geography, and duration (TX Bus & Comm §15.50),
        (2) No implied covenant of good faith and fair dealing in most commercial contracts,
        (3) Consequential damages waivers broadly enforced, (4) Attorneys' fees available to prevailing party
        under TX Civil Practice & Remedies Code §38.001.""",
        "metadata": {"type": "general", "jurisdiction": "texas", "source": "TX Business & Commerce Code"},
    },
    {
        "content": """NEW YORK CONTRACT LAW SPECIFICS: NY is the preferred governing law for commercial contracts.
        Key rules: (1) Statute of Frauds requires contracts >$500 for goods in writing (UCC 2-201),
        (2) Parol evidence rule strictly enforced with integration clauses,
        (3) Non-competes scrutinized under 'reasonableness' standard,
        (4) UCC gap-fillers apply to goods contracts, (5) CPLR §7515 limits mandatory arbitration
        in sexual harassment claims.""",
        "metadata": {"type": "general", "jurisdiction": "new_york", "source": "NY UCC + CPLR"},
    },
    {
        "content": """AUTO-RENEWAL RED FLAGS: (1) Auto-renewal with notice period shorter than 30 days,
        (2) Substantially higher rates upon renewal without notice, (3) No cap on renewal periods,
        (4) Different termination rights for renewal vs initial term, (5) California requires clear
        disclosure of auto-renewal terms (CA Business & Professions Code §17601-17606).
        New York General Obligations Law §5-903 requires written notice for auto-renewal in service contracts.""",
        "metadata": {"type": "renewal", "jurisdiction": "general", "source": "CA + NY Auto-Renewal Law"},
    },
    {
        "content": """FORCE MAJEURE STANDARD: Post-COVID, force majeure clauses are heavily scrutinized.
        Well-drafted clauses include: (1) Specific triggering events (pandemic, war, government action, etc.),
        (2) Notice requirements (typically 10-30 days), (3) Mitigation obligations,
        (4) Duration limits after which either party may terminate, (5) Payment obligations during FM period.
        Vague 'acts of God' language without specificity is a red flag.""",
        "metadata": {"type": "force_majeure", "jurisdiction": "general", "source": "Post-COVID Contract Standards"},
    },
]


class LegalKnowledgeBase:
    """FAISS-backed RAG knowledge base for legal standards."""

    def __init__(
        self,
        persist_path: str = "./faiss_legal_kb",
        embedding_provider: str = "ollama",
    ):
        self.persist_path = persist_path
        self.embeddings = get_embedding_model(provider=embedding_provider)
        self.vectorstore = None
        self._initialize()

    def _initialize(self):
        if Path(self.persist_path).exists():
            print("[KB] Loading existing FAISS index...")
            self.vectorstore = FAISS.load_local(
                self.persist_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            print("[KB] Building FAISS index from legal standards...")
            self._build_index()

    def _build_index(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

        docs = []
        for standard in LEGAL_STANDARDS:
            chunks = splitter.split_text(standard["content"])
            for chunk in chunks:
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata=standard["metadata"],
                    )
                )

        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.vectorstore.save_local(self.persist_path)
        print(f"[KB] Built index with {len(docs)} chunks, saved to {self.persist_path}")

    def retrieve(
        self,
        query: str,
        clause_type: str,
        jurisdiction: str,
        k: int = 3,
    ) -> list[Document]:
        results = self.vectorstore.similarity_search(query, k=k * 2)

        filtered = []
        for result in results:
            result_type = result.metadata.get("type")
            result_jurisdiction = result.metadata.get("jurisdiction", "general")

            if result_type not in (clause_type, "general"):
                continue
            if result_jurisdiction not in (jurisdiction, "general"):
                continue
            filtered.append(result)

        return filtered[:k] if filtered else results[:k]

    def add_custom_standards(self, documents: list[Document]):
        self.vectorstore.add_documents(documents)
        self.vectorstore.save_local(self.persist_path)
