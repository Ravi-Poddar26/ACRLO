"""participant_module/adapter.py

Generic, evidence-first TRL assessment module.

Pipeline: PDF text (page-aware) -> sentence windows -> hybrid retrieval (TF-IDF + MiniLM,
fused with reciprocal-rank fusion) -> NLI verification -> criterion status ->
TRL decision (ladder walk) -> contract-valid result.

Every supporting_span is cut from the cleaned text of the cited page, so it is a verbatim
substring of that page (whitespace-normalised). No report-specific code.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logging.getLogger("pypdf").setLevel(logging.ERROR)
log = logging.getLogger("apctt.module")

EMBED_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
)
NLI_MODEL = (
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c",
    "bddf8c5411c34ac3565e16e04384fd68b2618dda",
)

MAX_SPAN_CHARS = 450  # a "short quotation"
MAX_PAGES_PER_CRITERION = 3  # only the first 3 distinct pages are scored
TRL_ASSERTION = re.compile(r"\bTRLs?\b|technology readiness level", re.IGNORECASE)
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

# Evidence-type hints derived from the criterion text itself, so a changed checklist still works.
EXPANSIONS: list[tuple[str, str]] = [
    (r"principle|theor|physical basis|scientific",
     "basic principles theory physical basis governing equations scientific foundation"),
    (r"concept|formulat|practical application",
     "proposed concept approach design method intended application"),
    (r"proof|analytical|experimental|critical function",
     "experiment results demonstrated analysis simulation showed critical function"),
    (r"laborator|breadboard", "laboratory test experiment measured bench tested"),
    (r"relevant environment|simulated|relevant",
     "tested simulated environment field conditions representative"),
    (r"prototype", "prototype demonstrated tested built"),
    (r"operational|in operation|deployed",
     "deployed operational site in service used successfully field operation"),
    (r"qualif|requirement|acceptance",
     "qualified qualification acceptance test requirements specification verified"),
    (r"quantif|measured|performance|compared",
     "measured performance compared requirement threshold criterion target value results"),
    (r"test condition|configuration|test article|defined",
     "test setup conditions configuration test article described"),
    (r"actual technology|assessed|target",
     "the technology itself the system evaluated in this report"),
]


# --------------------------------------------------------------------------- text extraction


def extract_pages(pdf_path: Path, extractor: str = "pypdf") -> list[dict[str, Any]]:
    """Return [{'page': n, 'text': raw_text}] with 1-based page numbers."""
    texts: list[str] | None = None

    if extractor == "pypdfium2":
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(str(pdf_path))
            try:
                texts = [pdf[i].get_textpage().get_text_range() or "" for i in range(len(pdf))]
            finally:
                pdf.close()
        except Exception:  # noqa: BLE001 - fall back to pypdf
            log.warning("pypdfium2 failed on %s; falling back to pypdf", pdf_path)
            texts = None

    if texts is None:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 - one bad page must not fail the case
                texts.append("")

    return [{"page": i, "text": t} for i, t in enumerate(texts, start=1)]


def clean_text(raw: str) -> str:
    """Whitespace-normalise one page. Spans are cut from this text, so they always match it."""
    text = raw.replace("\u00ad", "")
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)  # re-join hyphenated line breaks
    return re.sub(r"\s+", " ", text).strip()


def classify_page(raw: str, clean: str) -> str:
    """content / contents / references / low_content (a retrieval signal)."""
    if len(clean.split()) < 30:
        return "low_content"
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    first = lines[:15]
    has_heading = any(re.fullmatch(r"contents", ln, re.IGNORECASE) for ln in first)
    leaders = sum(bool(re.search(r"\.{3,}", ln)) for ln in first)
    if has_heading and leaders >= 3:
        return "contents"
    ref_heading = re.compile(r"^(?:\d+\s+)?references(?: cited)?(?:\s+\d+)?$", re.IGNORECASE)
    if any(ref_heading.fullmatch(ln) for ln in first[:5]):
        return "references"
    return "content"


ABBREVIATIONS = {
    "u.s.", "e.g.", "i.e.", "al.", "fig.", "figs.", "no.", "vs.", "approx.", "dr.", "mr.",
    "st.", "inc.", "co.", "eq.", "ca.", "etc.", "cf.", "vol.", "sec.", "eqs.",
}
CAPTION_LEAD = re.compile(r"(?:Figure|Fig\.|Table|Plate)\s+[\dA-Z]+[a-z]?\.\s+[A-Z]")
FRONT_MATTER = re.compile(
    r"U\.S\. Department of the Interior|Suggested citation|Prepared in cooperation|ISSN|"
    r"Open-File Report|Scientific Investigations Report|Professional Paper|Contents",
    re.IGNORECASE,
)
JUNK_START = re.compile(
    r"^(?:photograph|photo|credit|source|image|courtesy|figure|fig\.|table|plate)\b", re.I
)
JUNK_ANYWHERE = re.compile(
    r"photograph by|photo credit|suggested citation|https?://|doi\.org|\bISSN\b|"
    r"U\.S\. Department of the Interior|Open-File Report|Scientific Investigations Report",
    re.IGNORECASE,
)
CITATION = re.compile(r"\((?:[^()]{0,60}?\b(?:19|20)\d\d[a-z]?)\)|\band others\b", re.IGNORECASE)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence offsets. Does not split after abbreviations (U.S., et al., Fig.) or after a
    caption lead such as 'Figure 7.' so captions stay in one piece and can be recognised."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_BREAK.finditer(text):
        head = text[start : match.start()]
        token = head.split()[-1].lower() if head.split() else ""
        if token in ABBREVIATIONS or re.fullmatch(r"(?:figure|fig\.|table|plate)\s*\d*[a-z]?\.", 
                                                    " ".join(head.split()[-2:]).lower()):
            continue
        spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if e - s >= 20]


def is_junk(span: str) -> bool:
    """Photo credits, captions, cover/citation text, reference-like or non-prose fragments."""
    if JUNK_START.match(span) or JUNK_ANYWHERE.search(span) or CAPTION_LEAD.search(span):
        return True
    if len(CITATION.findall(span)) >= 3:
        return True
    letters = sum(ch.isalpha() for ch in span)
    return letters < 0.6 * len(span)


def build_windows(
    document_id: str, pages: list[dict[str, Any]], size: int = 2, stride: int = 1
) -> list[dict[str, Any]]:
    """Sentence windows. Each window text is a verbatim substring of its page's clean text."""
    windows: list[dict[str, Any]] = []
    for page in pages:
        if page["page_type"] in ("contents", "references", "front_matter"):
            continue
        text = page["clean"]
        sentences = sentence_spans(text)
        for i in range(0, len(sentences), stride):
            group = sentences[i : i + size]
            start, end = group[0][0], group[-1][1]
            span = text[start : min(end, start + MAX_SPAN_CHARS)].strip()
            if len(span) < 40 or TRL_ASSERTION.search(span) or is_junk(span):
                continue  # a bare TRL claim is never valid evidence
            windows.append({"document_id": document_id, "page": page["page"], "text": span})
    return windows


# --------------------------------------------------------------------------- retrieval helpers


def expand_query(description: str) -> str:
    extra = [terms for pattern, terms in EXPANSIONS if re.search(pattern, description, re.I)]
    return " ".join([description, *extra])


def rrf(rank_lists: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal-rank fusion of several ranked index lists."""
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def model_dir(model: tuple[str, str]) -> Path:
    name, revision = model
    root = Path(os.environ["APCTT_MODEL_STORE"]) / name
    pinned = root / revision
    if pinned.is_dir():
        return pinned
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"model not found in store: {name}")
    return candidates[0]


def ladder_levels(criteria: list[Mapping[str, Any]]) -> dict[str, int]:
    """criterion_id -> TRL level for the maturity ladder (never hard-codes IDs).

    Uses an explicit level field if the checklist has one; otherwise the first nine criteria in
    order are levels 1..9 (the official checklist lists the ladder first). Fewer than nine
    criteria and no level field -> no ladder, so the module abstains instead of guessing.
    """
    explicit: dict[str, int] = {}
    for criterion in criteria:
        for key in ("trl", "trl_level", "level", "maturity_level"):
            value = criterion.get(key)
            if isinstance(value, int) and 1 <= value <= 9:
                explicit[criterion["criterion_id"]] = value
                break
    if explicit:
        return explicit
    if len(criteria) >= 9:
        return {c["criterion_id"]: i + 1 for i, c in enumerate(criteria[:9])}
    return {}

def decide_trl(levels: dict[str, int], status: dict[str, str]) -> int | None:
    if not levels:
        return None

    met_levels = {
        level
        for cid, level in levels.items()
        if status.get(cid) == "met"
    }

    if not met_levels:
        return None

    max_met = max(met_levels)

    not_met_levels = {
        level
        for cid, level in levels.items()
        if status.get(cid) == "not met"
    }

    for blocker in not_met_levels:
        if blocker <= max_met:
            max_met = min(max_met, blocker - 1)

    return max_met if max_met >= 1 else None


def make_threshold_fn(levels: dict[str, int], threshold: float, high_bonus: float):
    """Per-criterion acceptance threshold; levels 7-9 can be made stricter via high_bonus."""

    def threshold_for(criterion_id: str) -> float:
        bump = high_bonus if levels.get(criterion_id, 0) >= 7 else 0.0
        return min(threshold + bump, 0.98)

    return threshold_for


def assign_evidence(
    scored: dict[str, list[tuple[float, int]]],
    levels: dict[str, int],
    threshold_for: Any,
) -> dict[str, list[tuple[float, int]]]:
    """Accepted (score, window_index) per criterion, best first.

    A passage may support only ONE ladder criterion (the one it entails most strongly);
    criteria outside the ladder may share passages.
    """
    owner: dict[int, str] = {}
    owner_score: dict[int, float] = {}
    for cid in levels:
        limit = threshold_for(cid)
        for score, idx in scored.get(cid, []):
            if score >= limit and score > owner_score.get(idx, -1.0):
                owner_score[idx], owner[idx] = score, cid
    accepted: dict[str, list[tuple[float, int]]] = {}
    for cid, candidates in scored.items():
        limit = threshold_for(cid)
        accepted[cid] = sorted(
            (
                (score, idx)
                for score, idx in candidates
                if score >= limit and (cid not in levels or owner.get(idx) == cid)
            ),
            reverse=True,
        )
    return accepted


def trl_from_scores(
    scored: dict[str, list[tuple[float, int]]],
    levels: dict[str, int],
    threshold: float,
    high_bonus: float = 0.0,
) -> int | None:
    """TRL a given threshold would give (used by the offline sweep in batch_eval.py)."""
    accepted = assign_evidence(scored, levels, make_threshold_fn(levels, threshold, high_bonus))
    status = {cid: ("met" if accepted.get(cid) else "insufficient evidence") for cid in levels}
    return decide_trl(levels, status)


def left_ms(context: Any) -> float:
    fn = getattr(context, "remaining_ms", None)
    try:
        return float(fn()) if fn else 1e9
    except Exception:  # noqa: BLE001
        return 1e9


# --------------------------------------------------------------------------- the module


class AssessmentModule:
    def __init__(self) -> None:
        self.checklist: Mapping[str, Any] = {}
        self.settings: Mapping[str, Any] = {}
        self.device = "cpu"
        self.embedding_model: Any = None
        self.nli_tokenizer: Any = None
        self.nli_model: Any = None
        self.entail_idx = 0
        self.contra_idx: int | None = None
        self.debug: dict[str, Any] = {}

    def load(self, config: Mapping[str, Any], asset_root: Path) -> None:
        self.checklist = config["checklist"]
        self.settings = config["module"]["settings"] or {}
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        wanted = str(self.settings.get("device", "cpu"))
        self.device = wanted if wanted.startswith("cuda") and torch.cuda.is_available() else "cpu"

        self.embedding_model = SentenceTransformer(
            str(model_dir(EMBED_MODEL)), device=self.device, local_files_only=True
        )
        nli_path = model_dir(NLI_MODEL)
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_path, local_files_only=True)
        self.nli_model = (
            AutoModelForSequenceClassification.from_pretrained(
                nli_path, local_files_only=True, dtype="float32"
            )
            .to(self.device)
            .eval()
        )
        for key, label in self.nli_model.config.id2label.items():
            if str(label).lower().startswith("entail"):
                self.entail_idx = int(key)
            if str(label).lower().startswith("contradict"):
                self.contra_idx = int(key)
    # ---- NLI -------------------------------------------------------------------------------

    def _nli_scores(self, pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
        """Return (entailment_prob, contradiction_prob) per pair."""
        import torch

        results: list[tuple[float, float]] = []
        for i in range(0, len(pairs), 16):
            batch = pairs[i : i + 16]
            enc = self.nli_tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            ).to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self.nli_model(**enc).logits, dim=-1)
            for row in probs:
                e = float(row[self.entail_idx])
                c = float(row[self.contra_idx]) if self.contra_idx is not None else 0.0
                results.append((e, c))
        return results

    # ---- main ------------------------------------------------------------------------------

    def assess(self, case: Mapping[str, Any], context: Any) -> dict[str, Any]:
        try:
            return self._assess(case, context)
        except Exception:  # noqa: BLE001 - abstaining scores more than a failed case
            log.exception("assessment error; returning an abstained result")
            return self._result(case, {}, {}, {}, {}, None, "an internal error stopped the analysis")

    def _assess(self, case: Mapping[str, Any], context: Any) -> dict[str, Any]:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        cfg = self.settings
        threshold = float(cfg.get("nli_threshold", 0.55))
        n_candidates = int(cfg.get("candidates_per_criterion", 8))
        high_bonus = float(cfg.get("high_level_bonus", 0.0))
        max_windows = int(cfg.get("max_windows", 8000))
        extractor = str(cfg.get("extractor", "pypdf"))
        hypothesis_mode = str(cfg.get("hypothesis_mode", "plain"))
        target_text = " ".join(str(case["assessment_target"].get("description", "")).split())[:160]

        criteria = list(self.checklist["criteria"])
        ids = [c["criterion_id"] for c in criteria]
        levels = ladder_levels(criteria)

        threshold_for = make_threshold_fn(levels, threshold, high_bonus)
        status = {cid: "insufficient evidence" for cid in ids}
        evidence: dict[str, list[dict[str, Any]]] = {cid: [] for cid in ids}
        rationale: dict[str, str] = {}

        # 1. pages -> windows
        per_doc_pages: list[tuple[str, list[dict[str, Any]]]] = []
        for document in case["documents"]:
            pages = extract_pages(Path(document["path"]), extractor)
            for page in pages:
                page["clean"] = clean_text(page["text"])
                page["page_type"] = classify_page(page["text"], page["clean"])
                if page["page"] <= 6 and len(FRONT_MATTER.findall(page["clean"])) >= 2:
                    page["page_type"] = "front_matter"
            per_doc_pages.append((document["document_id"], pages))

        windows: list[dict[str, Any]] = []
        for stride in (1, 2, 3):
            windows = [
                w for doc_id, pages in per_doc_pages for w in build_windows(doc_id, pages, 2, stride)
            ]
            if len(windows) <= max_windows:
                break
        if not windows:
            return self._result(case, status, evidence, rationale, {}, None,
                                "no extractable text was found (scanned PDF?)")

        # 2. indexes
        texts = [w["text"] for w in windows]
        tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2))
        matrix = tfidf.fit_transform(texts)
        embeddings = None
        if self.embedding_model is not None and left_ms(context) > 70_000:
            embeddings = self.embedding_model.encode(
                texts, batch_size=64, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )

        # pre-built page_type lookup to avoid nested search inside criterion loop
        page_type_lookup: dict[tuple[str, int], str] = {
            (doc_id, page["page"]): page["page_type"]
            for doc_id, pages in per_doc_pages
            for page in pages
        }
        
        # 3. retrieve + verify per criterion
        scored: dict[str, list[tuple[float, int]]] = {}
        contra_scored: dict[str, list[tuple[float, int]]] = {}
        for criterion in criteria:
            if left_ms(context) < 15_000:
                log.warning("time budget nearly used; remaining criteria stay insufficient")
                break
            cid = criterion["criterion_id"]
            description = str(criterion["description"])
            query = expand_query(description)

            sparse = (matrix @ tfidf.transform([query]).T).toarray().ravel()
            lists = [[int(i) for i in np.argsort(-sparse)[:60] if sparse[i] > 0]]
            if embeddings is not None:
                q = self.embedding_model.encode(
                    [query], convert_to_numpy=True, normalize_embeddings=True,
                    show_progress_bar=False,
                )[0]
                lists.append([int(i) for i in np.argsort(-(embeddings @ q))[:60]])

            picked: list[int] = []
            per_page: dict[tuple[str, int], int] = {}
            for idx in rrf(lists):
                key = (windows[idx]["document_id"], windows[idx]["page"])
                if per_page.get(key, 0) >= 2:
                    continue
                per_page[key] = per_page.get(key, 0) + 1
                picked.append(idx)
                if len(picked) >= n_candidates:
                    break
            if not picked:
                continue
            picked.sort(key=lambda idx: {
                "content": 0, "low_content": 1,
                "contents": 2, "references": 3, "front_matter": 4
            }.get(
                page_type_lookup.get(
                    (windows[idx]["document_id"], windows[idx]["page"]), "content"
                ), 0
            ))    

            intended_env = str(case.get("intended_environment", "")).lower().strip()[:80]
            intended_app = str(case.get("intended_application", "")).lower().strip()[:80]
            context_prefix = target_text
            if intended_env:
                context_prefix += f" in {intended_env}"
                
            hypothesis = f"For {context_prefix}: {description}"
            
            nli_results = self._nli_scores([(windows[i]["text"], hypothesis) for i in picked])
            scored[cid] = [(e, idx) for (e, _c), idx in zip(nli_results, picked, strict=True)]
            contra_scored[cid] = [(c, idx) for (_e, c), idx in zip(nli_results, picked, strict=True)]

        # 4. assign evidence (one passage supports at most one ladder criterion)
        accepted_map = assign_evidence(scored, levels, threshold_for)
        self.debug = {"scored": scored, "levels": levels, "windows": windows}

        for cid in ids:
            candidates = scored.get(cid, [])
            accepted = accepted_map.get(cid, [])
            seen: set[tuple[str, int]] = set()
            for score, idx in accepted:  # best first; one span per page; at most 3 pages
                w = windows[idx]
                key = (w["document_id"], w["page"])
                if key in seen:
                    continue
                seen.add(key)
                evidence[cid].append(
                    {"document_id": w["document_id"], "page": w["page"],
                     "supporting_span": w["text"]}
                )
                if len(seen) >= MAX_PAGES_PER_CRITERION:
                    break
            if evidence[cid]:
                status[cid] = "met"
                best_score, best_idx = accepted[0]
                rationale[cid] = (
                    f"Best passage on page {windows[best_idx]['page']} "
                    f"(entailment {best_score:.2f})."
                )
            elif candidates:
                best_score, best_idx = max(candidates)
                rationale[cid] = (
                    f"Not established: best candidate on page {windows[best_idx]['page']} "
                    f"scored entailment {best_score:.2f} (threshold {threshold_for(cid):.2f})."
                )

        # 4b. Mark "not met" where contradiction is strong and no evidence was found
        CONTRA_THRESHOLD = 0.65
        for cid in ids:
            if status[cid] == "insufficient evidence" and cid in contra_scored:
                candidates = contra_scored[cid]
                if candidates:
                    best_contra, best_idx = max(candidates)
                    if best_contra >= CONTRA_THRESHOLD:
                        status[cid] = "not met"
                        rationale[cid] = (
                            rationale.get(cid, "")
                            + f" Contradiction score {best_contra:.2f} on page "
                            f"{windows[best_idx]['page']} suggests criterion is not satisfied."
                        ).strip()

        # 5. TRL
        trl = decide_trl(levels, status) if levels else None
        reason = None if trl else "no ladder criterion was supported by verified evidence"
        return self._result(case, status, evidence, rationale, {}, trl, reason)

    # ---- result ----------------------------------------------------------------------------

    def _result(
        self,
        case: Mapping[str, Any],
        status: dict[str, str],
        evidence: dict[str, list[dict[str, Any]]],
        rationale: dict[str, str],
        _unused: dict[str, Any],
        trl: int | None,
        abstain_note: str | None,
    ) -> dict[str, Any]:
        ids = [c["criterion_id"] for c in self.checklist["criteria"]]
        criterion_status = {cid: status.get(cid, "insufficient evidence") for cid in ids}
        missing = {
            cid: "No verified passage directly supports this criterion."
            for cid in ids
            if criterion_status[cid] != "met"
        }
        result: dict[str, Any] = {
            "schema_version": case["schema_version"],
            "checklist_version": self.checklist["version"],
            "case_id": case["case_id"],
            "assessment_target": dict(case["assessment_target"]),
            "assessment_status": "completed" if trl else "abstained",
            "predicted_trl": trl,
            "criterion_status": criterion_status,
            "evidence_reference": {cid: evidence.get(cid, []) for cid in ids},
            "missing_evidence": missing,
            "abstention_reason": None if trl else "insufficient evidence",
            "error_code": None,
        }
        if rationale:
            result["criterion_rationale"] = rationale
        if trl:
            result["trl_rationale"] = (
                f"Highest ladder level reached with verified evidence: TRL {trl}."
            )
        return result

    def reset_case(self) -> None:
        pass

    def close(self) -> None:
        pass