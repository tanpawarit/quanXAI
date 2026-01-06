"""SQLAlchemy models for query history and feedback."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class QueryHistory(Base):
    """Model for storing query history."""
    
    __tablename__ = "query_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=True)
    agents_used = Column(JSON, default=list)  # List of agent names
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to feedback
    feedbacks = relationship("Feedback", back_populates="query_history")
    
    def __repr__(self):
        return f"<QueryHistory(id={self.id}, query='{self.query[:30]}...')>"


class Feedback(Base):
    """Model for storing user feedback."""
    
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey("query_history.id"), nullable=False)
    rating = Column(Integer, nullable=False)  # 1-5
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to query
    query_history = relationship("QueryHistory", back_populates="feedbacks")
    
    def __repr__(self):
        return f"<Feedback(id={self.id}, query_id={self.query_id}, rating={self.rating})>"
