"""Lightweight, deterministic intent router for the FinSight AI assistant.

    User question
        -> route_intent()            (deterministic keyword routing, traceable)
        -> the matching backend tool (agent_tools.py; pure, verified math)
        -> a verified natural-language answer built ONLY from the tool facts
        -> (optionally) NVIDIA NIM explains those same verified facts

No LangChain/LangGraph or multi-agent framework is used — a small, explicit
if/elif router is enough to make tool selection explicit and demonstrable,
per the project's "keep it simple" requirement. The LLM is never given the
freedom to pick a tool or invent a number; routing and math are both
deterministic and backend-controlled.
"""
from __future__ import annotations

from . import agent_tools as tools

fmt = tools.fmt


def route_intent(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ('unusual', 'anomal', 'flagged', 'suspicious', 'review')):
        return 'UNUSUAL_TRANSACTIONS'
    if any(x in q for x in ('recurring', 'subscription')):
        return 'RECURRING_PAYMENTS'
    if any(x in q for x in ('compare', 'increase', 'decrease', 'change', 'trend', 'versus', ' vs ')):
        return 'SPENDING_COMPARISON'
    if any(x in q for x in ('search', 'find transactions', 'list transactions', 'show transactions')):
        return 'TRANSACTION_SEARCH'
    category = tools.detect_category(question)
    if category and any(x in q for x in ('how much', 'spent', 'spend')):
        return 'CATEGORY_SPENDING'
    if any(x in q for x in ('highest', 'most', 'where did')):
        return 'CATEGORY_SPENDING'
    return 'FINANCIAL_SUMMARY'


def run(question: str, rows: list) -> tuple:
    """Returns (answer, facts, tool_used) — all backend-verified."""
    q = question.lower()
    intent = route_intent(question)
    period = 'week' if any(x in q for x in ('this week', 'past week', 'last 7 days')) else 'month'

    if intent == 'UNUSUAL_TRANSACTIONS':
        facts = tools.unusual_transaction_tool(rows)
        if facts['flagged_count']:
            details = '; '.join(
                f"{fmt(t['amount'])} at {t['merchant']}: {', '.join(t['reasons'])}"
                for t in facts['flagged_transactions']
            )
            answer = f"{facts['flagged_count']} unusual transaction(s) need review. {details}"
        else:
            answer = 'No unusual transactions currently need review.'
        return answer, facts, 'unusual_transaction_tool'

    if intent == 'RECURRING_PAYMENTS':
        facts = tools.recurring_payment_tool(rows, period=period)
        if facts['recurring_payments']:
            answer = 'Recurring payments: ' + ', '.join(
                f"{p['merchant']} ({fmt(p['amount'])})" for p in facts['recurring_payments']
            )
        else:
            answer = 'No recurring payments identified.'
        return answer, facts, 'recurring_payment_tool'

    if intent == 'SPENDING_COMPARISON':
        category = tools.detect_category(question)
        facts = tools.spending_comparison_tool(rows, category=category)
        label = f" on {category}" if category else ''
        answer = (
            f"Spending{label} changed {facts['change_pct']:+.1f}% from last month "
            f"({fmt(facts['current_period'])} versus {fmt(facts['previous_period'])})."
        )
        return answer, facts, 'spending_comparison_tool'

    if intent == 'TRANSACTION_SEARCH':
        category = tools.detect_category(question)
        facts = tools.transaction_search_tool(rows, category=category, period=period)
        answer = f"Found {facts['count']} matching transaction(s)."
        return answer, facts, 'transaction_search_tool'

    if intent == 'CATEGORY_SPENDING':
        category = tools.detect_category(question)
        facts = tools.category_spending_tool(rows, category=category, period=period)
        if category:
            answer = f"You spent {fmt(facts['requested_category_amount'])} on {category} {facts['period']}."
        elif facts['top_category']:
            answer = f"Your highest spending category {facts['period']} is {facts['top_category']} at {fmt(facts['top_category_amount'])}."
        else:
            answer = f"No spending recorded {facts['period']}."
        return answer, facts, 'category_spending_tool'

    # FINANCIAL_SUMMARY (default / week / average / income percentage)
    facts = tools.financial_summary_tool(rows, period=period)
    if any(x in q for x in ('percentage', 'percent', '%')) and 'income' in q:
        answer = (
            f"You spent {fmt(facts['spending'])} out of {fmt(facts['income'])} income "
            f"{facts['period']}: {facts['income_share_pct']:.1f}% of your income."
        )
    elif 'average' in q:
        answer = f"Your average outgoing transaction {facts['period']} is {fmt(facts['average_transaction'])}."
    elif period == 'week':
        answer = f"You spent {fmt(facts['spending'])} across {facts['transaction_count']} transaction(s) this week."
    else:
        answer = (
            f"{facts['period'].title()}: spending is {fmt(facts['spending'])}, income is {fmt(facts['income'])}, "
            f"and estimated savings are {fmt(facts['estimated_savings'])}. Ask about percentages, categories, "
            f"this week, recurring payments, comparisons, or unusual transactions."
        )
    return answer, facts, 'financial_summary_tool'
