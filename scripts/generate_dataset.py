import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'backend'))
from app.services import seed_transactions
out=Path(__file__).resolve().parents[1]/'dataset'/'sample';out.mkdir(parents=True,exist_ok=True)
rows=seed_transactions()
with open(out/'transactions.csv','w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(f'Generated {len(rows)} synthetic transactions at {out / "transactions.csv"}')
