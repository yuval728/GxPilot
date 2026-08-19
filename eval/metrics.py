"""Task-specific metrics for GxP structured outputs."""
import re
from typing import Dict, List
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def exact_match(pred: str, target: str, ignore_case: bool = True, ignore_whitespace: bool = True) -> float:
    """Exact match for structured sections (e.g., deviation report fields)."""
    if ignore_case:
        pred = pred.lower()
        target = target.lower()
    if ignore_whitespace:
        pred = re.sub(r'\s+', ' ', pred).strip()
        target = re.sub(r'\s+', ' ', target).strip()
    return 1.0 if pred == target else 0.0


def section_exact_match(pred: str, target: str, section_headers: List[str]) -> Dict[str, float]:
    """Extract and compare specific sections by header (e.g., **Description:**, **Root Cause:**)."""
    results = {}
    for header in section_headers:
        pred_section = _extract_section(pred, header)
        target_section = _extract_section(target, header)
        results[header] = exact_match(pred_section, target_section)
    return results


def _extract_section(text: str, header: str) -> str:
    """Extract text between header and next header or end."""
    pattern = rf"{re.escape(header)}(.*?)(?=\*\*[A-Za-z ]+:\*\*|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def rouge_l(pred: str, target: str) -> Dict[str, float]:
    """ROUGE-L F1, precision, recall."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(target, pred)
    return {
        'rougeL_f1': scores['rougeL'].fmeasure,
        'rougeL_precision': scores['rougeL'].precision,
        'rougeL_recall': scores['rougeL'].recall,
    }


def bleu_score(pred: str, target: str) -> float:
    """Sentence-level BLEU with smoothing."""
    smoothie = SmoothingFunction().method4
    pred_tokens = pred.split()
    target_tokens = [target.split()]
    return sentence_bleu(target_tokens, pred_tokens, smoothing_function=smoothie)


def compute_all_metrics(pred: str, target: str, section_headers: List[str] = None) -> Dict:
    """Compute all metrics for a single prediction."""
    metrics = {
        'exact_match': exact_match(pred, target),
        'rougeL': rouge_l(pred, target),
        'bleu': bleu_score(pred, target),
    }
    if section_headers:
        metrics['section_match'] = section_match(pred, target, section_headers)
    return metrics


# Default section headers for GxP document types
DEVIATION_HEADERS = [
    "**Description:**", "**Immediate Action:**", "**Product Impact:**",
    "**Preliminary Classification:**", "**Next Step:**"
]
CAPA_HEADERS = [
    "**Root Cause**", "**Corrective Action:**", "**Preventive Action:**",
    "**Effectiveness Check:**", "**Target Date:**"
]
SOP_HEADERS = ["**SOP Section", "**Step:**", "**Escalation**"]
AUDIT_HEADERS = []