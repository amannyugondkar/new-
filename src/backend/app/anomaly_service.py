"""Explainable, dynamic anomaly detection for FinSight AI.

This module is the SINGLE SOURCE OF TRUTH for anomaly scoring. Every consumer
(dashboard metrics, the /api/anomalies endpoint, transaction listing, and the
assistant's "unusual transaction" tool) must go through `annotate_transactions`
so that scores/reasons never diverge between features.

Design (kept intentionally simple for an academic prototype):

    Transaction rows
        -> per-customer chronological feature engineering
        -> Isolation Forest (model signal, unusual-behaviour only, NOT fraud)
        -> deterministic behavioural rules (amount deviation, new merchant,
           new location, unusual hour) computed against each customer's own
           prior history
        -> weighted combination into a single 0.0-1.0 anomaly_score
        -> human-readable reasons generated only from signals that actually
           fired (never static/pre-written text)
        -> status + review_recommended flag for the human-in-the-loop workflow

Nothing here blocks, rejects, or autonomously acts on a transaction. It only
produces a score, a status, and reasons for a human analyst/admin to review.
"""
from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev
from typing import Iterable, Sequence

import numpy as np
from sklearn.ensemble import IsolationForest

# --- Score thresholds (documented per the spec) ---------------------------
# 0.00-0.39 -> NORMAL, 0.40-0.59 -> MONITOR, 0.60-1.00 -> UNUSUAL (review
# recommended). These are simple, explainable cut points rather than a
# black-box classifier decision boundary.
NORMAL_UPPER = 0.40
MONITOR_UPPER = 0.60
REVIEW_THRESHOLD = MONITOR_UPPER  # kept as a separate name for readability

# --- Combined score weights (documented, simple linear fusion) ------------
# model: Isolation Forest's opinion of how unusual the transaction's feature
#        vector is relative to the rest of the dataset.
# amount_deviation: how many "customer-normal" standard deviations away the
#        amount is from that customer's own historical DEBIT amounts.
# new_merchant / new_location / unusual_hour: transparent behavioural flags
#        computed from each customer's own transaction history (not a global
#        hardcoded rule such as "amount > 10000" or "2am = anomaly").
WEIGHTS = {
    'model': 0.40,
    'amount_deviation': 0.25,
    'new_merchant': 0.15,
    'new_location': 0.10,
    'unusual_hour': 0.10,
}

MIN_HISTORY_FOR_HOUR_PROFILE = 5   # need a few data points before judging "unusual hour"
MIN_ROWS_FOR_MODEL = 8             # below this, IsolationForest is not statistically meaningful
RANDOM_STATE = 42                  # reproducible/deterministic model behaviour

_CACHE: dict = {}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _dataset_signature(rows: Sequence) -> tuple:
    """Cheap signature so we don't retrain the Isolation Forest on every
    request when the underlying transaction set hasn't changed."""
    return tuple(
        (r.id, r.customer_id, round(float(r.amount), 4), r.transaction_type,
         r.merchant, r.location, r.timestamp.isoformat())
        for r in sorted(rows, key=lambda r: r.id)
    )


def _per_customer_chronological(rows: Sequence) -> dict:
    by_customer: dict = {}
    for r in rows:
        by_customer.setdefault(r.customer_id, []).append(r)
    for cust in by_customer:
        by_customer[cust].sort(key=lambda r: r.timestamp)
    return by_customer


def _engineer_features(rows: Sequence):
    """Compute per-transaction behavioural signals using only that customer's
    OWN transaction history that occurred strictly before the transaction
    (chronologically), so "new merchant"/"new location"/"unusual hour" are
    genuinely about novelty rather than data leakage from the future."""
    by_customer = _per_customer_chronological(rows)
    per_row = {}
    for cust, txns in by_customer.items():
        for idx, t in enumerate(txns):
            history = txns[:idx]
            # Category-specific baseline: comparing a coffee purchase against a
            # rent payment baseline would produce meaningless deviations, so we
            # baseline each DEBIT against the customer's own history for the
            # SAME category, falling back to the customer's overall DEBIT
            # history only when there isn't enough category-specific history yet.
            same_category_history = [h.amount for h in history if h.transaction_type == 'DEBIT' and h.category == t.category]
            all_debit_history = [h.amount for h in history if h.transaction_type == 'DEBIT']
            baseline_pool = same_category_history if len(same_category_history) >= 2 else all_debit_history

            if t.transaction_type == 'DEBIT' and baseline_pool:
                baseline_mean = mean(baseline_pool)
                baseline_std = pstdev(baseline_pool) if len(baseline_pool) > 1 else 0.0
                if baseline_std <= 0:
                    # Not enough variance to compute a z-score; fall back to a
                    # conservative proportion of the mean instead of dividing by 0.
                    baseline_std = max(baseline_mean * 0.15, 1.0)
                # Only amounts ABOVE the customer's own baseline are treated as
                # a deviation signal here — spending unusually LESS than normal
                # is not the behaviour this signal is meant to catch.
                z = max((t.amount - baseline_mean) / baseline_std, 0.0)
            else:
                baseline_mean = mean(baseline_pool) if baseline_pool else t.amount
                z = 0.0
            amount_deviation_component = _clip01(z / 4.0)  # >=4 std devs above baseline -> fully anomalous

            # --- new merchant/recipient (needs at least some prior history) ---
            merchants_seen = {h.merchant for h in history}
            new_merchant = 1 if history and t.merchant not in merchants_seen else 0

            # --- new location ---
            locations_seen = {h.location for h in history}
            new_location = 1 if history and t.location not in locations_seen else 0

            # --- unusual hour, derived from the customer's own activity profile ---
            hours_seen = [h.timestamp.hour for h in history]
            if len(hours_seen) >= MIN_HISTORY_FOR_HOUR_PROFILE:
                counts = Counter(hours_seen)
                freq = counts.get(t.timestamp.hour, 0) / len(hours_seen)
                unusual_hour = 1 if freq < 0.05 else 0
            else:
                unusual_hour = 0

            per_row[t.id] = {
                'z': z,
                'amount_deviation_component': amount_deviation_component,
                'baseline_mean': baseline_mean,
                'new_merchant': new_merchant,
                'new_location': new_location,
                'unusual_hour': unusual_hour,
            }
    return per_row


def _model_components(rows: Sequence, features: dict) -> dict:
    """Fit a lightweight Isolation Forest across the (small) dataset and
    convert its output into a 0-1 "how unusual is this feature vector"
    component. Skipped entirely (component=0) when there isn't enough data
    to be statistically meaningful, per the small-dataset guidance."""
    if len(rows) < MIN_ROWS_FOR_MODEL:
        return {r.id: 0.0 for r in rows}

    ordered = list(rows)
    matrix = []
    for r in ordered:
        f = features[r.id]
        matrix.append([
            float(r.amount),
            float(r.timestamp.hour),
            float(r.timestamp.weekday()),
            float(f['z']),
            float(f['new_merchant']),
            float(f['new_location']),
            float(f['unusual_hour']),
            1.0 if r.transaction_type == 'DEBIT' else 0.0,
        ])
    X = np.array(matrix)
    model = IsolationForest(n_estimators=200, contamination='auto', random_state=RANDOM_STATE)
    model.fit(X)
    raw = -model.score_samples(X)  # higher raw == more anomalous
    lo, hi = raw.min(), raw.max()
    if hi - lo <= 1e-9:
        normalized = np.zeros_like(raw)
    else:
        normalized = (raw - lo) / (hi - lo)
    return {r.id: _clip01(v) for r, v in zip(ordered, normalized)}


def _status_for(score: float) -> str:
    if score < NORMAL_UPPER:
        return 'NORMAL'
    if score < MONITOR_UPPER:
        return 'MONITOR'
    return 'UNUSUAL'


def _reasons_for(row, feat, model_component, combined_score, status) -> list:
    if status == 'NORMAL':
        return []
    reasons = []
    if feat['amount_deviation_component'] >= 0.5:
        reasons.append(
            f"Amount (₹{row.amount:,.0f}) is significantly above this customer's typical "
            f"transaction amount (baseline ≈ ₹{feat['baseline_mean']:,.0f})."
        )
    if feat['new_merchant']:
        reasons.append("Merchant/recipient has not appeared in this customer's recent transaction history.")
    if feat['new_location']:
        reasons.append("Transaction occurred in a location not seen in this customer's recent transaction history.")
    if feat['unusual_hour']:
        reasons.append("Transaction occurred during a time of day this customer rarely transacts.")
    if not reasons and model_component >= 0.6:
        reasons.append("Transaction pattern differs from this customer's typical behaviour based on model-based analysis.")
    if not reasons:
        reasons.append("Minor deviation from typical behaviour observed; below the review threshold.")
    return reasons


def _compute(rows: Sequence) -> dict:
    features = _engineer_features(rows)
    model_components = _model_components(rows, features)
    results = {}
    for r in rows:
        feat = features[r.id]
        model_component = model_components[r.id]
        combined = (
            WEIGHTS['model'] * model_component
            + WEIGHTS['amount_deviation'] * feat['amount_deviation_component']
            + WEIGHTS['new_merchant'] * feat['new_merchant']
            + WEIGHTS['new_location'] * feat['new_location']
            + WEIGHTS['unusual_hour'] * feat['unusual_hour']
        )
        combined = _clip01(combined)
        status = _status_for(combined)
        reasons = _reasons_for(r, feat, model_component, combined, status)
        results[r.id] = {
            'anomaly_score': round(combined, 4),
            'status': status,
            'review_recommended': status == 'UNUSUAL',
            'reasons': reasons,
            'reasons_text': '; '.join(reasons),
        }
    return results


def evaluate_transactions(rows: Iterable) -> dict:
    """Return {transaction_id: result} for the given rows. Cached by a cheap
    signature of the dataset so repeated requests against an unchanged
    dataset don't retrain the Isolation Forest every time."""
    rows = list(rows)
    if not rows:
        return {}
    sig = _dataset_signature(rows)
    cached = _CACHE.get(sig)
    if cached is not None:
        return cached
    result = _compute(rows)
    _CACHE[sig] = result
    return result


def annotate_transactions(rows: Iterable) -> list:
    """Mutate (in-memory only, never committed) each row with fresh,
    consistently-computed anomaly fields. This is the single choke point
    every feature (dashboard, anomalies endpoint, transaction listing,
    assistant tool) should call so there is one anomaly source of truth."""
    rows = list(rows)
    results = evaluate_transactions(rows)
    for r in rows:
        res = results.get(r.id)
        if not res:
            continue
        r.anomaly_score = res['anomaly_score']
        r.anomaly_reasons = res['reasons_text']
        # Additive, backward-compatible fields (new attributes, nothing removed).
        r.anomaly_status = res['status']
        r.review_recommended = res['review_recommended']
    return rows
