import os
from sqlalchemy import create_engine, String, Integer, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from .core import DATA_DIR
url=os.getenv('DATABASE_URL', f'sqlite:///{DATA_DIR / "finsight.db"}')
engine=create_engine(url, connect_args={'check_same_thread':False} if url.startswith('sqlite') else {})
SessionLocal=sessionmaker(bind=engine, autoflush=False)
class Base(DeclarativeBase): pass
class User(Base):
    __tablename__='users'; id:Mapped[int]=mapped_column(Integer,primary_key=True); email:Mapped[str]=mapped_column(String,unique=True); password_hash:Mapped[str]=mapped_column(String); role:Mapped[str]=mapped_column(String); customer_id:Mapped[str|None]=mapped_column(String,nullable=True)
class Transaction(Base):
    __tablename__='transactions'; id:Mapped[int]=mapped_column(Integer,primary_key=True); customer_id:Mapped[str]=mapped_column(String,index=True); timestamp:Mapped[DateTime]=mapped_column(DateTime); amount:Mapped[float]=mapped_column(Float); transaction_type:Mapped[str]=mapped_column(String); merchant:Mapped[str]=mapped_column(String); description:Mapped[str]=mapped_column(Text); category:Mapped[str]=mapped_column(String); confidence:Mapped[float]=mapped_column(Float); location:Mapped[str]=mapped_column(String); balance:Mapped[float]=mapped_column(Float); anomaly_score:Mapped[float]=mapped_column(Float,default=0); anomaly_reasons:Mapped[str]=mapped_column(Text,default=''); review_status:Mapped[str]=mapped_column(String,default='OPEN')
