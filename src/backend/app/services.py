import json
import logging
import os
from datetime import datetime, timedelta
from collections import defaultdict
import httpx
import numpy as np

from . import agent_router
from .anomaly_service import REVIEW_THRESHOLD

logger = logging.getLogger(__name__)
NVIDIA_NIM_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'
DEFAULT_NIM_MODEL = 'nvidia/nemotron-3-super-120b-a12b'
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

TRAIN=[('swiggy food order','Food'),('zomato dinner','Food'),('amazon pay shopping','Shopping'),('flipkart purchase','Shopping'),('hpcl fuel','Transport'),('uber ride','Transport'),('monthly rent','Housing'),('netflix subscription','Entertainment'),('electricity bill','Utilities'),('salary credit','Income'),('upi transfer','Transfer'),('mutual fund investment','Investment'),('hospital pharmacy','Healthcare')]
def classifier(): return make_pipeline(TfidfVectorizer(ngram_range=(1,2)),LogisticRegression(max_iter=400)).fit([x for x,y in TRAIN],[y for x,y in TRAIN])
MODEL=classifier()
def categorize(text):
    p=MODEL.predict_proba([text])[0]; i=int(np.argmax(p)); return str(MODEL.classes_[i]),round(float(p[i]),2)

def seed_transactions(customer='CUST001'):
    # NOTE: anomaly_score/anomaly_reasons are intentionally left at their
    # column defaults (0 / '') here. They are no longer predetermined or
    # manually injected — they are computed dynamically by anomaly_service
    # (see main.py's transactions_for -> annotate_transactions) every time
    # transactions are read. The single transfer below is still crafted to
    # be behaviourally atypical (much larger amount, brand-new recipient,
    # new location, off-hours) purely so the dynamic pipeline has a genuine
    # unusual pattern to detect — the SCORE itself is never hardcoded.
    rows=[]; balance=92000.; now=datetime.now().replace(hour=10,minute=0,second=0,microsecond=0)
    patterns=[('salary credit','Employer','Income',65000,'CREDIT'),('monthly rent','Green Homes','Housing',18000,'DEBIT'),('swiggy food order','Swiggy','Food',460,'DEBIT'),('amazon pay shopping','Amazon','Shopping',1250,'DEBIT'),('hpcl fuel','HPCL','Transport',2100,'DEBIT'),('netflix subscription','Netflix','Entertainment',649,'DEBIT'),('electricity bill','BESCOM','Utilities',1300,'DEBIT'),('upi transfer','Rahul','Transfer',1800,'DEBIT')]
    for month in range(2,-1,-1):
      for index,(desc,merchant,cat,amount,kind) in enumerate(patterns):
       dt=now-timedelta(days=month*30-index*2); val=amount*(1.22 if month==1 and cat=='Food' else 1); balance += val if kind=='CREDIT' else -val
       rows.append(dict(customer_id=customer,timestamp=dt,amount=val,transaction_type=kind,merchant=merchant,description=desc,category=cat,confidence=.91,location='Bengaluru',balance=balance,anomaly_score=0.0,anomaly_reasons='',review_status='OPEN'))
    dt=now-timedelta(days=2); balance-=38500
    rows.append(dict(customer_id=customer,timestamp=dt.replace(hour=2),amount=38500,transaction_type='DEBIT',merchant='New Recipient',description='upi transfer',category='Transfer',confidence=.93,location='Mumbai',balance=balance,anomaly_score=0.0,anomaly_reasons='',review_status='OPEN'))
    return rows

def metrics(ts):
    now=datetime.now(); cur=[t for t in ts if t.timestamp.month==now.month and t.timestamp.year==now.year]; prev=[t for t in ts if t.timestamp.month==(now-timedelta(days=30)).month]
    inc=sum(t.amount for t in cur if t.transaction_type=='CREDIT'); exp=sum(t.amount for t in cur if t.transaction_type=='DEBIT'); prevexp=sum(t.amount for t in prev if t.transaction_type=='DEBIT')
    cats=defaultdict(float)
    for t in cur:
      if t.transaction_type=='DEBIT': cats[t.category]+=t.amount
    return {'monthly_income':round(inc,2),'monthly_expenditure':round(exp,2),'estimated_savings':round(inc-exp,2),'current_balance':round(ts[-1].balance if ts else 0,2),'average_transaction':round(float(np.mean([t.amount for t in ts])) if ts else 0,2),'transaction_frequency':len(cur),'category_spending':dict(cats),'spending_change_pct':round(((exp-prevexp)/prevexp*100) if prevexp else 0,1),'unusual_count':sum(t.anomaly_score>=REVIEW_THRESHOLD for t in ts)}

# --- Agentic assistant --------------------------------------------------
# User question -> agent_router.run() picks a deterministic intent + backend
# tool -> verified facts + a verified answer string -> optional NVIDIA NIM
# explanation layer over those SAME verified facts (never independent math).

def _deterministic_grounded_assistant(question, ts):
    answer, facts, tool_used = agent_router.run(question, ts)
    return answer, 'deterministic verified analytics', facts, tool_used


def _nim_fallback_reason(exc):
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        detail = ''
        try:
            detail = exc.response.json().get('detail', '')
        except Exception:
            detail = exc.response.text[:160]
        if status == 401:
            return 'deterministic fallback (NVIDIA NIM key rejected)'
        if status == 410:
            return 'deterministic fallback (NVIDIA NIM model retired — update NVIDIA_NIM_MODEL)'
        if status == 404:
            return 'deterministic fallback (NVIDIA NIM model not enabled for your account)'
        return f'deterministic fallback (NVIDIA NIM HTTP {status})'
    if isinstance(exc, httpx.TimeoutException):
        return 'deterministic fallback (NVIDIA NIM request timed out)'
    return 'deterministic fallback (NVIDIA NIM unavailable)'

def _explain_with_nim(question, answer, facts, api_key):
    model = os.getenv('NVIDIA_NIM_MODEL', DEFAULT_NIM_MODEL).strip() or DEFAULT_NIM_MODEL
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': 'You are FinSight AI. Explain only the verified facts supplied. Give a complete, friendly 3-5 sentence answer: state the result, compare it with the relevant baseline when present, explain what the change means in plain language, and offer one practical non-investment budgeting or review suggestion. Never alter numbers, infer missing facts, guarantee outcomes, independently calculate any financial value, or call an unusual transaction fraud.'},
            {'role': 'user', 'content': f'Customer question: {question}\nVerified answer: {answer}\nVerified facts: {json.dumps(facts)}'},
        ],
        'temperature': 0.1,
        'max_tokens': 420,
        'chat_template_kwargs': {'thinking': False},
    }
    with httpx.Client(timeout=45.0) as client:
        response = client.post(
            NVIDIA_NIM_URL,
            json=payload,
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        )
        response.raise_for_status()
        message = response.json()['choices'][0]['message']
    content = (message.get('content') or '').strip()
    if not content:
        raise ValueError('Empty model response')
    if content.lower().startswith("here's a thinking process"):
        raise ValueError('Model returned reasoning output instead of a final answer')
    return content

def grounded_assistant(question, ts):
    answer, fallback_provider, facts, tool_used = _deterministic_grounded_assistant(question, ts)
    api_key = os.getenv('NVIDIA_NIM_API_KEY', '').strip()
    if not api_key:
        return answer, 'deterministic fallback (NVIDIA NIM key not configured)', facts, tool_used
    try:
        content = _explain_with_nim(question, answer, facts, api_key)
        return content, 'NVIDIA NIM grounded explanation', facts, tool_used
    except Exception as exc:
        logger.warning('NVIDIA NIM explanation failed: %s', exc)
        return answer, _nim_fallback_reason(exc), facts, tool_used
