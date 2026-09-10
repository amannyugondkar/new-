import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'backend'))
from app.main import startup
startup(); print('Database initialized with demo users and transactions.')
