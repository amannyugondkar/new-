import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
from jose import jwt
from passlib.context import CryptContext

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)
load_dotenv(ROOT / '.env')
SECRET = os.getenv('JWT_SECRET', 'development-secret-change-me')
pwd = CryptContext(schemes=['bcrypt'], deprecated='auto')
class Role(str, Enum): CUSTOMER='CUSTOMER'; BANK_ANALYST='BANK_ANALYST'; ADMIN='ADMIN'
def hash_password(value: str): return pwd.hash(value)
def verify_password(value: str, digest: str): return pwd.verify(value, digest)
def token_for(user_id: int, role: str):
    return jwt.encode({'sub': str(user_id), 'role': role, 'exp': datetime.now(timezone.utc)+timedelta(hours=8)}, SECRET, algorithm='HS256')
