from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Resolve project root and place DB in the top-level `database/` folder,
# unless DATABASE_URL points somewhere else.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_DIR = os.path.join(project_root, "database")
DB_PATH = os.path.join(DB_DIR, "backend.db")
SQLALCHEMY_DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")
if "DATABASE_URL" not in os.environ:
    os.makedirs(DB_DIR, exist_ok=True)

connect_args = {}
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
