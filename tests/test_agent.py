import os
os.environ.setdefault('DATABASE_URL', 'sqlite:///./data/test_agent.db')
from datetime import datetime, timedelta
from types import SimpleNamespace

from src.backend.app import agent_router, agent_tools


def _txn(id, dt, amount, ttype, merchant, category, anomaly_score=0.0, anomaly_reasons='',
         anomaly_status='NORMAL', review_status='OPEN', description=''):
    return SimpleNamespace(
        id=id, customer_id='CUST001', timestamp=dt, amount=amount, transaction_type=ttype,
        merchant=merchant, category=category, location='Bengaluru', description=description,
        anomaly_score=anomaly_score, anomaly_reasons=anomaly_reasons,
        anomaly_status=anomaly_status, review_status=review_status,
    )


def _sample_rows():
    now = datetime.now()
    rows = [
        _txn(1, now.replace(day=1, hour=9), 65000, 'CREDIT', 'Employer', 'Income', description='salary credit'),
        _txn(2, now.replace(day=2, hour=9), 18000, 'DEBIT', 'Green Homes', 'Housing', description='monthly rent'),
        _txn(3, now.replace(day=3, hour=13), 500, 'DEBIT', 'Swiggy', 'Food', description='swiggy food order'),
        _txn(4, now.replace(day=4, hour=13), 700, 'DEBIT', 'Swiggy', 'Food', description='swiggy food order'),
        _txn(5, now.replace(day=5, hour=2), 38500, 'DEBIT', 'New Recipient', 'Transfer',
             anomaly_score=0.95, anomaly_status='UNUSUAL', anomaly_reasons='Amount is unusually high; New recipient',
             description='upi transfer'),
    ]
    return rows


def test_intent_routing_unusual():
    assert agent_router.route_intent('Show unusual transactions') == 'UNUSUAL_TRANSACTIONS'


def test_intent_routing_category():
    assert agent_router.route_intent('How much did I spend on food?') == 'CATEGORY_SPENDING'


def test_intent_routing_comparison():
    assert agent_router.route_intent('Compare my spending to last month') == 'SPENDING_COMPARISON'


def test_intent_routing_default_summary():
    assert agent_router.route_intent('Tell me about my finances') == 'FINANCIAL_SUMMARY'


def test_tool_selection_and_verified_facts_for_category():
    rows = _sample_rows()
    answer, facts, tool_used = agent_router.run('How much did I spend on food?', rows)
    assert tool_used == 'category_spending_tool'
    assert facts['requested_category'] == 'Food'
    assert facts['requested_category_amount'] == 1200.0
    assert '1,200' in answer or '1200' in answer


def test_spending_comparison_tool_facts():
    rows = _sample_rows()
    answer, facts, tool_used = agent_router.run('Compare my spending to last month', rows)
    assert tool_used == 'spending_comparison_tool'
    assert 'current_period' in facts and 'previous_period' in facts


def test_unusual_transaction_tool_reads_annotated_scores_only():
    rows = _sample_rows()
    answer, facts, tool_used = agent_router.run('Show unusual transactions', rows)
    assert tool_used == 'unusual_transaction_tool'
    assert facts['flagged_count'] == 1
    assert facts['flagged_transactions'][0]['id'] == 5
    assert 'fraud' not in answer.lower()


def test_financial_summary_tool_facts():
    rows = _sample_rows()
    answer, facts, tool_used = agent_router.run('Give me a summary of my finances', rows)
    assert tool_used == 'financial_summary_tool'
    assert facts['income'] == 65000.0


def test_recurring_payment_tool():
    rows = _sample_rows()
    facts = agent_tools.recurring_payment_tool(rows)
    assert any(p['merchant'] == 'Green Homes' for p in facts['recurring_payments'])


def test_transaction_search_tool_is_filter_based_only():
    rows = _sample_rows()
    facts = agent_tools.transaction_search_tool(rows, category='Food')
    assert facts['count'] == 2
    assert all(t['category'] == 'Food' for t in facts['transactions'])


def test_grounded_assistant_fallback_without_nim_key():
    os.environ['NVIDIA_NIM_API_KEY'] = ''
    from src.backend.app.services import grounded_assistant
    rows = _sample_rows()
    answer, provider, facts, tool_used = grounded_assistant('Show unusual transactions', rows)
    assert 'deterministic' in provider.lower()
    assert tool_used == 'unusual_transaction_tool'
    assert isinstance(facts, dict)
