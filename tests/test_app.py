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
