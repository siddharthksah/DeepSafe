"""
DeepSafe Database Module
SQLAlchemy ORM models and database session management for storing analysis history.
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./deepsafe_history.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class AnalysisHistory(Base):
    """Store analysis results for auditing and reporting."""
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, index=True, nullable=True)  # User who ran the analysis
    media_type = Column(String, nullable=False)  # image, video, audio
    media_name = Column(String, nullable=True)
    
    # Analysis Results
    verdict = Column(String, nullable=False)  # "real" or "fake"
    confidence = Column(Float, nullable=False)
    ensemble_method = Column(String, nullable=False)  # "stacking", "voting", "average"
    ensemble_score = Column(Float, nullable=False)
    
    # Metadata
    inference_time = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Optional: Store full JSON response
    full_response = Column(Text, nullable=True)

    def __repr__(self):
        return f"<AnalysisHistory(id={self.id}, request_id={self.request_id}, verdict={self.verdict})>"


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for FastAPI routes to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
