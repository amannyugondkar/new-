from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from jose import JWTError, jwt
from sqlalchemy import select
from .db import Base,engine,SessionLocal,User,Transaction
from .core import Role,hash_password,verify_password,token_for,SECRET
from .services import seed_transactions,metrics,grounded_assistant,categorize
app=FastAPI(title='FinSight AI'); app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:3000'],allow_credentials=True,allow_methods=['*'],allow_headers=['*']); oauth=OAuth2PasswordBearer(tokenUrl='/api/auth/login')
def db():
 s=SessionLocal()
 try: yield s
 finally:s.close()
def current(token=Depends(oauth),s=Depends(db)):
 try: u=s.get(User,int(jwt.decode(token,SECRET,algorithms=['HS256'])['sub']))
 except (JWTError,KeyError):u=None
 if not u: raise HTTPException(401,'Invalid authentication')
 return u
def require(*roles):
 def check(u=Depends(current)):
  if u.role not in roles: raise HTTPException(403,'Insufficient role')
  return u
 return check
class Login(BaseModel):email:str; password:str
class Ask(BaseModel):question:str
class Review(BaseModel):action:str
@app.on_event('startup')
def startup():
 Base.metadata.create_all(engine)
 with SessionLocal() as s:
  if not s.scalar(select(User).limit(1)):
   s.add_all([User(email='customer@finsight.local',password_hash=hash_password('Demo@123'),role='CUSTOMER',customer_id='CUST001'),User(email='analyst@finsight.local',password_hash=hash_password('Demo@123'),role='BANK_ANALYST',customer_id=None),User(email='admin@finsight.local',password_hash=hash_password('Demo@123'),role='ADMIN',customer_id=None)])
   s.add_all([Transaction(**x) for x in seed_transactions()]);s.commit()
@app.get('/health')
def health():return {'status':'ok'}
@app.post('/api/auth/login')
def login(x:Login,s=Depends(db)):
 u=s.scalar(select(User).where(User.email==x.email))
 if not u or not verify_password(x.password,u.password_hash):raise HTTPException(401,'Invalid email or password')
 return {'access_token':token_for(u.id,u.role),'token_type':'bearer','role':u.role}
@app.get('/api/auth/me')
def me(u=Depends(current)):return {'email':u.email,'role':u.role,'customer_id':u.customer_id}
def transactions_for(u,s): return list(s.scalars(select(Transaction).where(Transaction.customer_id==u.customer_id).order_by(Transaction.timestamp))) if u.role=='CUSTOMER' else list(s.scalars(select(Transaction).order_by(Transaction.timestamp)))
@app.get('/api/dashboard')
def dashboard(u=Depends(current),s=Depends(db)):return metrics(transactions_for(u,s))
@app.get('/api/profile')
def profile(u=Depends(current),s=Depends(db)):return metrics(transactions_for(u,s))
@app.get('/api/transactions')
def transactions(u=Depends(current),s=Depends(db)):return transactions_for(u,s)
@app.get('/api/anomalies')
def anomalies(u=Depends(current),s=Depends(db)):return [t for t in transactions_for(u,s) if t.anomaly_score>=.6]
@app.post('/api/anomalies/{transaction_id}/review')
def review(transaction_id:int,x:Review,u=Depends(require('BANK_ANALYST','ADMIN')),s=Depends(db)):
 t=s.get(Transaction,transaction_id)
 if not t:raise HTTPException(404,'Transaction not found')
 t.review_status=x.action.upper();s.commit();return {'id':t.id,'review_status':t.review_status}
@app.post('/api/assistant/query')
def ask(x:Ask,u=Depends(current),s=Depends(db)):
 answer, provider, facts = grounded_assistant(x.question, transactions_for(u,s))
 return {'answer':answer,'grounded':True,'provider':provider,'facts':facts}
@app.post('/api/categorize')
def classify(description:str,u=Depends(current)): c,p=categorize(description);return {'category':c,'confidence':p}
