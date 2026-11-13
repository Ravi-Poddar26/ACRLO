# TRL assessment module

An evidence-first Assessment Package for the APCTT Hackathon 2026. It reads the PDF reports of
one case and returns, for every checklist criterion, `met` / `insufficient evidence`, the page
evidence with a short verbatim quotation, and a TRL (or an abstention).

## How it works
1. **Text extraction**: page-aware text with `pypdf` (optionally `pypdfium2`). Each page is
   whitespace-cleaned; quotations are cut from that cleaned text, so every `supporting_span` is
   a verbatim substring of the cited page.
2. **Passages**: two-sentence windows, abbreviation-aware. Cover/title pages, contents and
   reference pages, photo credits, figure/table captions, URLs and citation-heavy fragments are
   not used as evidence, and neither is any sentence that merely states a TRL.
3. **Retrieval**: TF-IDF (scikit-learn) and `all-MiniLM-L6-v2` embeddings, fused by reciprocal
   rank fusion. Queries are built from each criterion's own description plus evidence-type hints
   derived from its wording, so a changed checklist works without code changes.
4. **Verification**: the NLI model `deberta-v3-base-zeroshot-v2.0-c` scores whether a passage
   entails the criterion. A criterion is `met` at or above `nli_threshold`. A passage supports
   at most one maturity-ladder criterion (the one it entails most strongly).
5. **TRL**: the highest met ladder criterion (lower levels are implied). If no ladder
   criterion is met the module abstains with `insufficient evidence`.
6. **Safety**: any internal error returns a contract-valid abstained result; the module
   checks `context.remaining_ms()` and skips work when time runs short.

## Models
Catalogue models only, loaded offline from `APCTT_MODEL_STORE`: `all-MiniLM-L6-v2` and
`deberta-v3-base-zeroshot-v2.0-c`. No weights are shipped; the module runs on the CPU.

## Settings (`submission.json` -> `settings`)
`nli_threshold` (default 0.55), `high_level_bonus` (extra threshold for levels 7-9, default 0),
`candidates_per_criterion`, `max_windows`, `extractor` (`pypdf` or `pypdfium2`),
`hypothesis_mode` (`plain` or `target`).

## Known limitations
No OCR for scanned pages; no separate `not met` detection (unproven criteria are reported as
`insufficient evidence`); thresholds were tuned on a small number of public sample cases.

## Licence
MIT. Third-party components are listed in `participant_module/THIRD_PARTY_NOTICES.md`.# ACRLO
