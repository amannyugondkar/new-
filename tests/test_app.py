import os
os.environ['DATABASE_URL']='sqlite:///./data/test.db'
from fastapi.testclient import TestClient
from src.backend.app.main import app
def test_flow():
 with TestClient(app) as c:
  r=c.post('/api/auth/login',json={'email':'customer@finsight.local','password':'Demo@123'});assert r.status_code==200
  h={'Authorization':'Bearer '+r.json()['access_token']}; d=c.get('/api/dashboard',headers=h);assert d.status_code==200 and d.json()['monthly_income']>0
  assert c.post('/api/assistant/query',json={'question':'How much food?'},headers=h).json()['grounded']
def test_role_protection():
 with TestClient(app) as c:
  t=c.post('/api/auth/login',json={'email':'customer@finsight.local','password':'Demo@123'}).json()['access_token']; assert c.post('/api/anomalies/1/review',json={'action':'reviewed'},headers={'Authorization':'Bearer '+t}).status_code==403

def test_customer_transaction_isolation():
 with TestClient(app) as c:
  t=c.post('/api/auth/login',json={'email':'customer@finsight.local','password':'Demo@123'}).json()['access_token']
  rows=c.get('/api/transactions',headers={'Authorization':'Bearer '+t}).json()
  assert rows and all(r['customer_id']=='CUST001' for r in rows)

def test_analyst_review_workflow_still_works():
 with TestClient(app) as c:
  a=c.post('/api/auth/login',json={'email':'analyst@finsight.local','password':'Demo@123'}).json()['access_token']
  h={'Authorization':'Bearer '+a}
  anomalies=c.get('/api/anomalies',headers=h).json()
  assert len(anomalies)>=1
  rid=anomalies[0]['id']
  r=c.post(f'/api/anomalies/{rid}/review',json={'action':'reviewed'},headers=h)
  assert r.status_code==200 and r.json()['review_status']=='REVIEWED'

def test_categorization_still_functional():
 with TestClient(app) as c:
  t=c.post('/api/auth/login',json={'email':'customer@finsight.local','password':'Demo@123'}).json()['access_token']
  r=c.post('/api/categorize?description=swiggy%20food%20order',headers={'Authorization':'Bearer '+t})
  assert r.status_code==200 and r.json()['category']=='Food'

def test_dashboard_and_anomaly_counts_are_consistent():
 with TestClient(app) as c:
  t=c.post('/api/auth/login',json={'email':'customer@finsight.local','password':'Demo@123'}).json()['access_token']
  h={'Authorization':'Bearer '+t}
  dashboard=c.get('/api/dashboard',headers=h).json()
  anomalies=c.get('/api/anomalies',headers=h).json()
  assert dashboard['unusual_count']==len(anomalies)

def test_assistant_response_includes_tool_metadata():
 with TestClient(app) as c:
  t=c.post('/api/auth/login',json={'email':'customer@finsight.local','password':'Demo@123'}).json()['access_token']
  r=c.post('/api/assistant/query',json={'question':'Show unusual transactions'},headers={'Authorization':'Bearer '+t})
  body=r.json()
  assert body['grounded'] is True
  assert 'tool_used' in body and body['tool_used']=='unusual_transaction_tool'
  assert 'fraud' not in body['answer'].lower()
