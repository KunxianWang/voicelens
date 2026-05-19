from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from voicelens.nlp.absa.schema import ONTOLOGY_CODES_LATEST


def _aspects(row: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = row.get(key) or []
    return value if isinstance(value, list) else []


def _by_code(aspects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for aspect in aspects:
        code = aspect.get("aspect_code")
        if isinstance(code, str) and code not in out:
            out[code] = aspect
    return out


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({x for pair in pairs for x in pair})
    n = len(pairs)
    observed = sum(1 for gold, pred in pairs if gold == pred) / n
    gold_counts = Counter(gold for gold, _ in pairs)
    pred_counts = Counter(pred for _, pred in pairs)
    expected = sum((gold_counts[label] / n) * (pred_counts[label] / n) for label in labels)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return _safe_div(observed - expected, 1 - expected)


def _eval_codes(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    """Per-aspect breakdown should cover every code that could appear in
    gold OR predictions for this holdout. Start from the latest ontology
    so absent aspects still show up with zero counts, then add any
    out-of-ontology codes that did show up (so the breakdown reflects
    reality instead of silently hiding them).
    """
    seen: set[str] = set(ONTOLOGY_CODES_LATEST)
    for row in rows:
        for key in ("gold_aspects", "predicted_aspects"):
            for aspect in _aspects(row, key):
                code = aspect.get("aspect_code") if isinstance(aspect, dict) else None
                if isinstance(code, str):
                    seen.add(code)
    return tuple(
        sorted(seen, key=lambda c: (c not in ONTOLOGY_CODES_LATEST, c))
    )


def evaluate_absa_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    codes = _eval_codes(rows)
    per_aspect: dict[str, Counter[str]] = {code: Counter() for code in codes}
    sentiment_pairs: list[tuple[str, str]] = []
    severity_pairs: list[tuple[str, str]] = []
    evidence_total = 0
    evidence_verbatim = 0
    invalid_rows = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        text = str(row.get("text_raw") or "")
        gold = _by_code(_aspects(row, "gold_aspects"))
        pred = _by_code(_aspects(row, "predicted_aspects"))
        if row.get("validation_errors"):
            invalid_rows += 1

        for code in codes:
            in_gold = code in gold
            in_pred = code in pred
            if in_gold and in_pred:
                per_aspect[code]["tp"] += 1
                gold_sent = str(gold[code].get("sentiment"))
                pred_sent = str(pred[code].get("sentiment"))
                sentiment_pairs.append((gold_sent, pred_sent))
                confusion[code][f"{gold_sent}->{pred_sent}"] += 1
                if gold_sent == "negative":
                    gold_sev = str(gold[code].get("severity"))
                    pred_sev = str(pred[code].get("severity"))
                    severity_pairs.append((gold_sev, pred_sev))
            elif in_gold:
                per_aspect[code]["fn"] += 1
                confusion[code]["gold_present->missing"] += 1
            elif in_pred:
                per_aspect[code]["fp"] += 1
                confusion[code]["spurious_prediction"] += 1

        for aspect in pred.values():
            quote = aspect.get("evidence_quote")
            if isinstance(quote, str) and quote:
                evidence_total += 1
                if quote in text:
                    evidence_verbatim += 1

    tp = sum(c["tp"] for c in per_aspect.values())
    fp = sum(c["fp"] for c in per_aspect.values())
    fn = sum(c["fn"] for c in per_aspect.values())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    aspect_metrics: dict[str, dict[str, float | int]] = {}
    f1s: list[float] = []
    for code, counts in per_aspect.items():
        p = _safe_div(counts["tp"], counts["tp"] + counts["fp"])
        r = _safe_div(counts["tp"], counts["tp"] + counts["fn"])
        code_f1 = _safe_div(2 * p * r, p + r)
        f1s.append(code_f1)
        aspect_metrics[code] = {
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(code_f1, 4),
        }

    sentiment_correct = sum(1 for gold, pred in sentiment_pairs if gold == pred)
    severity_correct = sum(1 for gold, pred in severity_pairs if gold == pred)
    return {
        "n_reviews": len(rows),
        "aspect_extraction": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "micro_f1": round(f1, 4),
            "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            "per_aspect": aspect_metrics,
        },
        "sentiment_accuracy": round(_safe_div(sentiment_correct, len(sentiment_pairs)), 4),
        "sentiment_cohens_kappa": (
            None if _cohen_kappa(sentiment_pairs) is None else round(_cohen_kappa(sentiment_pairs), 4)
        ),
        "severity_accuracy_on_negative_mentions": round(
            _safe_div(severity_correct, len(severity_pairs)), 4
        ),
        "evidence_quote_verbatim_rate": round(_safe_div(evidence_verbatim, evidence_total), 4),
        "invalid_output_rate": round(_safe_div(invalid_rows, len(rows)), 4),
        "per_aspect_confusion_summary": {
            code: dict(counter) for code, counter in sorted(confusion.items())
        },
    }
