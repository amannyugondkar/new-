"""Deterministic backend "tools" for the agentic financial assistant.

Every tool takes the customer's (already anomaly-annotated) transaction rows
and returns plain, verified, backend-calculated facts. Tools never call an
LLM and never estimate/guess a number — all math happens here so the
explanation layer (NVIDIA NIM) only ever explains numbers that already exist.

Anomaly facts are retrieved via `anomaly_service`-annotated rows only, never
recomputed independently, so the assistant, dashboard, and anomalies endpoint
always agree.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

CATEGORY_LOOKUP = {
    'food': 'Food', 'shopping': 'Shopping', 'transport': 'Transport',
    'housing': 'Housing', 'health': 'Healthcare', 'entertainment': 'Entertainment',
    'utilities': 'Utilities', 'transfer': 'Transfer', 'investment': 'Investment',
}

fmt = lambda n: f'₹{n:,.0f}'


def _debits(rows):
    return [t for t in rows if t.transaction_type == 'DEBIT']


def _period_rows(rows, period: str):
    now = datetime.now()
    if period == 'week':
        return [t for t in rows if t.timestamp >= now - timedelta(days=7)]
    if period == 'previous_month':
        prev = now.replace(day=1) - timedelta(days=1)
        return [t for t in rows if t.timestamp.year == prev.year and t.timestamp.month == prev.month]
    # default: current month
    return [t for t in rows if t.timestamp.year == now.year and t.timestamp.month == now.month]


def detect_category(question: str):
    q = question.lower()
    return next((v for k, v in CATEGORY_LOOKUP.items() if k in q), None)


def financial_summary_tool(rows, period='month'):
    period_rows = _period_rows(rows, period)
    spend = sum(t.amount for t in _debits(period_rows))
    income = sum(t.amount for t in period_rows if t.transaction_type == 'CREDIT')
    debit_count = len(_debits(period_rows))
    return {
        'period': 'this week' if period == 'week' else 'this month',
        'spending': round(spend, 2),
        'income': round(income, 2),
        'estimated_savings': round(income - spend, 2),
        'transaction_count': len(period_rows),
        'average_transaction': round(spend / debit_count, 2) if debit_count else 0.0,
        'income_share_pct': round(spend / income * 100, 1) if income else 0.0,
    }


def category_spending_tool(rows, category=None, period='month'):
    period_rows = _period_rows(rows, period)
    cats = defaultdict(float)
    for t in _debits(period_rows):
        cats[t.category] += t.amount
    top = max(cats, key=cats.get) if cats else None
    return {
        'period': 'this week' if period == 'week' else 'this month',
        'category_spending': dict(cats),
        'top_category': top,
        'top_category_amount': round(cats[top], 2) if top else 0.0,
        'requested_category': category,
        'requested_category_amount': round(cats.get(category, 0.0), 2) if category else None,
    }


def spending_comparison_tool(rows, category=None):
    now = datetime.now()
    current_rows = _period_rows(rows, 'month')
    previous_rows = _period_rows(rows, 'previous_month')
    if category:
        current = sum(t.amount for t in current_rows if t.transaction_type == 'DEBIT' and t.category == category)
        previous = sum(t.amount for t in previous_rows if t.transaction_type == 'DEBIT' and t.category == category)
    else:
        current = sum(t.amount for t in _debits(current_rows))
        previous = sum(t.amount for t in _debits(previous_rows))
    pct = round((current - previous) / previous * 100, 1) if previous else 0.0
    return {
        'category': category,
        'current_period': round(current, 2),
        'previous_period': round(previous, 2),
        'change_pct': pct,
    }


def transaction_search_tool(rows, category=None, merchant=None, transaction_type=None, period=None):
    """Safe, filter-based lookup over the customer's own transactions.
    No arbitrary SQL/query strings are ever accepted from the assistant."""
    results = rows
    if period:
        results = _period_rows(results, period)
    if category:
        results = [t for t in results if t.category == category]
    if merchant:
        results = [t for t in results if merchant.lower() in t.merchant.lower()]
    if transaction_type:
        results = [t for t in results if t.transaction_type == transaction_type.upper()]
    return {
        'count': len(results),
        'transactions': [
            {
                'id': t.id, 'amount': t.amount, 'merchant': t.merchant,
                'category': t.category, 'transaction_type': t.transaction_type,
                'timestamp': t.timestamp.isoformat(),
            }
            for t in results[:10]
        ],
    }


def unusual_transaction_tool(rows):
    """Reads anomaly facts from already-annotated rows (single source of
    truth is anomaly_service; this tool never recomputes a score itself)."""
    flagged = [t for t in rows if getattr(t, 'anomaly_score', 0) >= 0.6]
    return {
        'flagged_count': len(flagged),
        'flagged_transactions': [
            {
                'id': t.id,
                'amount': t.amount,
                'merchant': t.merchant,
                'anomaly_score': t.anomaly_score,
                'status': getattr(t, 'anomaly_status', 'UNUSUAL'),
                'review_status': t.review_status,
                'reasons': t.anomaly_reasons.split('; ') if t.anomaly_reasons else [],
            }
            for t in flagged
        ],
    }


def recurring_payment_tool(rows, period='month'):
    period_rows = _period_rows(rows, period)
    recurring = [
        t for t in _debits(period_rows)
        if any(w in t.description.lower() for w in ('rent', 'subscription', 'bill'))
    ]
    return {
        'period': 'this week' if period == 'week' else 'this month',
        'recurring_count': len(recurring),
        'recurring_payments': [
            {'merchant': t.merchant, 'amount': t.amount, 'category': t.category}
            for t in recurring
        ],
    }
