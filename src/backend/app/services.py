import json
import logging
import os
from datetime import datetime, timedelta
from collections import defaultdict
import httpx
import numpy as np

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
    rows=[]; balance=92000.; now=datetime.now().replace(hour=10,minute=0,second=0,microsecond=0)
    patterns=[('salary credit','Employer','Income',65000,'CREDIT'),('monthly rent','Green Homes','Housing',18000,'DEBIT'),('swiggy food order','Swiggy','Food',460,'DEBIT'),('amazon pay shopping','Amazon','Shopping',1250,'DEBIT'),('hpcl fuel','HPCL','Transport',2100,'DEBIT'),('netflix subscription','Netflix','Entertainment',649,'DEBIT'),('electricity bill','BESCOM','Utilities',1300,'DEBIT'),('upi transfer','Rahul','Transfer',1800,'DEBIT')]
    for month in range(2,-1,-1):
      for index,(desc,merchant,cat,amount,kind) in enumerate(patterns):
       dt=now-timedelta(days=month*30-index*2); val=amount*(1.22 if month==1 and cat=='Food' else 1); balance += val if kind=='CREDIT' else -val
       rows.append(dict(customer_id=customer,timestamp=dt,amount=val,transaction_type=kind,merchant=merchant,description=desc,category=cat,confidence=.91,location='Bengaluru',balance=balance,anomaly_score=.04,anomaly_reasons='',review_status='OPEN'))
    dt=now-timedelta(days=2); balance-=38500
    rows.append(dict(customer_id=customer,timestamp=dt.replace(hour=2),amount=38500,transaction_type='DEBIT',merchant='New Recipient',description='upi transfer',category='Transfer',confidence=.93,location='Mumbai',balance=balance,anomaly_score=.91,anomaly_reasons='Amount is substantially above normal transfer range; New recipient; Occurred outside normal active hours; New location',review_status='OPEN'))
    return rows
def metrics(ts):
    now=datetime.now(); cur=[t for t in ts if t.timestamp.month==now.month and t.timestamp.year==now.year]; prev=[t for t in ts if t.timestamp.month==(now-timedelta(days=30)).month]
    inc=sum(t.amount for t in cur if t.transaction_type=='CREDIT'); exp=sum(t.amount for t in cur if t.transaction_type=='DEBIT'); prevexp=sum(t.amount for t in prev if t.transaction_type=='DEBIT')
    cats=defaultdict(float)
    for t in cur:
      if t.transaction_type=='DEBIT': cats[t.category]+=t.amount
    return {'monthly_income':round(inc,2),'monthly_expenditure':round(exp,2),'estimated_savings':round(inc-exp,2),'current_balance':round(ts[-1].balance if ts else 0,2),'average_transaction':round(float(np.mean([t.amount for t in ts])) if ts else 0,2),'transaction_frequency':len(cur),'category_spending':dict(cats),'spending_change_pct':round(((exp-prevexp)/prevexp*100) if prevexp else 0,1),'unusual_count':sum(t.anomaly_score>=.6 for t in ts)}
def assistant(question, m):
    q=question.lower(); cats=m['category_spending']; top=max(cats,key=cats.get) if cats else 'no category'
    if 'food' in q: return f"Verified result: you spent ₹{cats.get('Food',0):,.0f} on Food this month."
    if 'unusual' in q or 'anomal' in q: return f"Verified result: {m['unusual_count']} unusual transaction(s) need review."
    if 'most' in q or 'where' in q: return f"Verified result: {top} is your highest spending category at ₹{cats.get(top,0):,.0f} this month."
    if 'compare' in q or 'increase' in q: return f"Verified result: monthly spending changed {m['spending_change_pct']:+.1f}% from the previous month."
    if 'average' in q: return f"Verified result: average transaction amount is ₹{m['average_transaction']:,.0f}."
    return f"Verified result: monthly expenditure is ₹{m['monthly_expenditure']:,.0f}; income is ₹{m['monthly_income']:,.0f}. Ask about food, highest spend, unusual transactions, or comparison."

def grounded_assistant(question, ts):
    q=question.lower(); now=datetime.now(); week=[t for t in ts if t.timestamp>=now-timedelta(days=7)]; month=[t for t in ts if t.timestamp.year==now.year and t.timestamp.month==now.month]
    debits=lambda rows:[t for t in rows if t.transaction_type=='DEBIT']; fmt=lambda n:f'₹{n:,.0f}'
    rows=week if any(x in q for x in ('this week','past week','last 7 days')) else month; label='this week' if rows is week else 'this month'; spend=sum(t.amount for t in debits(rows)); income=sum(t.amount for t in rows if t.transaction_type=='CREDIT')
    cats=defaultdict(float)
    for t in debits(rows):cats[t.category]+=t.amount
    lookup={'food':'Food','shopping':'Shopping','transport':'Transport','housing':'Housing','health':'Healthcare','entertainment':'Entertainment','utilities':'Utilities','transfer':'Transfer','investment':'Investment'}; category=next((v for k,v in lookup.items() if k in q),None)
    if any(x in q for x in ('percentage','percent','%')) and 'income' in q: answer=f'You spent {fmt(spend)} out of {fmt(income)} income {label}: {spend/income*100 if income else 0:.1f}% of your income.'
    elif category and any(x in q for x in ('how much','spent','spend')): answer=f'You spent {fmt(cats[category])} on {category} {label}.'
    elif rows is week: answer=f'You spent {fmt(spend)} across {len(debits(rows))} outgoing transaction(s) this week.'
    elif any(x in q for x in ('recurring','subscription')):
      r=[t for t in debits(rows) if any(w in t.description.lower() for w in ('rent','subscription','bill'))]; answer='Recurring payments: '+(', '.join(f'{t.merchant} ({fmt(t.amount)})' for t in r) if r else 'No recurring payments identified.')
    elif any(x in q for x in ('unusual','anomal','flagged')):
      r=[t for t in ts if t.anomaly_score>=.6]; answer=f'{len(r)} unusual transaction(s) need review. '+('; '.join(f'{fmt(t.amount)} at {t.merchant}: {t.anomaly_reasons}' for t in r))
    elif any(x in q for x in ('highest','most','where did')):
      top=max(cats,key=cats.get) if cats else None; answer=f'Your highest spending category {label} is {top} at {fmt(cats[top])}.' if top else f'No spending recorded {label}.'
    elif any(x in q for x in ('compare','increase','decrease','change')):
      prev=now.replace(day=1)-timedelta(days=1); prior=sum(t.amount for t in ts if t.transaction_type=='DEBIT' and t.timestamp.year==prev.year and t.timestamp.month==prev.month); pct=(spend-prior)/prior*100 if prior else 0; answer=f'Spending changed {pct:+.1f}% from last month ({fmt(spend)} versus {fmt(prior)}).'
    elif 'average' in q: answer=f'Your average outgoing transaction {label} is {fmt(spend/len(debits(rows)) if debits(rows) else 0)}.'
    else: answer=f'{label.title()}: spending is {fmt(spend)}, income is {fmt(income)}, and estimated savings are {fmt(income-spend)}. Ask about percentages, categories, this week, recurring payments, comparisons, or unusual transactions.'
    return answer,'deterministic verified analytics',{'period':label,'spending':spend,'income':income,'category_spending':dict(cats)}

# Optional NVIDIA NIM layer: only the already-calculated response and structured facts leave the app.
_deterministic_grounded_assistant = grounded_assistant

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
            {'role': 'system', 'content': 'You are FinSight AI. Explain only the verified facts supplied. Give a complete, friendly 3-5 sentence answer: state the result, compare it with the relevant baseline when present, explain what the change means in plain language, and offer one practical non-investment budgeting or review suggestion. Never alter numbers, infer missing facts, guarantee outcomes, or call an unusual transaction fraud.'},
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
    answer, fallback_provider, facts = _deterministic_grounded_assistant(question, ts)
    api_key = os.getenv('NVIDIA_NIM_API_KEY', '').strip()
    if not api_key:
        return answer, 'deterministic fallback (NVIDIA NIM key not configured)', facts
    try:
        content = _explain_with_nim(question, answer, facts, api_key)
        return content, 'NVIDIA NIM grounded explanation', facts
    except Exception as exc:
        logger.warning('NVIDIA NIM explanation failed: %s', exc)
        return answer, _nim_fallback_reason(exc), facts

def richer_verified_assistant(question, ts):
    q=question.lower(); now=datetime.now(); current=[t for t in ts if t.timestamp.year==now.year and t.timestamp.month==now.month]; previous_date=now.replace(day=1)-timedelta(days=1); previous=[t for t in ts if t.timestamp.year==previous_date.year and t.timestamp.month==previous_date.month]
    lookup={'food':'Food','shopping':'Shopping','transport':'Transport','housing':'Housing','health':'Healthcare','entertainment':'Entertainment','utilities':'Utilities','transfer':'Transfer','investment':'Investment'}; category=next((v for k,v in lookup.items() if k in q),None); fmt=lambda n:f'₹{n:,.0f}'
    if category and any(x in q for x in ('lately','more','less','increase','decrease','change','compare','trend')):
        cur=sum(t.amount for t in current if t.transaction_type=='DEBIT' and t.category==category); prev=sum(t.amount for t in previous if t.transaction_type=='DEBIT' and t.category==category); pct=(cur-prev)/prev*100 if prev else 0
        direction='more' if pct>0 else 'less' if pct<0 else 'the same amount'
        recommendation=(f' To keep this category in view, consider setting a monthly {category.lower()} budget and reviewing the next purchase before it is made.' if pct>0 else ' This category is stable or lower; keep monitoring it alongside your overall monthly budget.')
        answer=f'You spent {fmt(cur)} on {category} this month versus {fmt(prev)} last month. That is {abs(pct):.1f}% {direction}.'+recommendation
        return answer,'deterministic verified analytics',{'category':category,'current_month':cur,'previous_month':prev,'change_pct':round(pct,1)}
    return _original_local_assistant(question, ts)

_original_local_assistant = _deterministic_grounded_assistant
_deterministic_grounded_assistant = richer_verified_assistant
