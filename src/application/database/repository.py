"""Database repository for query history and feedback."""

from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker, Session

from src.config import settings
from src.application.database.models import Base, QueryHistory, Feedback


class DatabaseSession:
    """Database session manager."""
    
    def __init__(self, database_url: str = None):
        """
        Initialize database session.
        
        Args:
            database_url: Database URL (default from settings)
        """
        self._url = database_url or settings.database_url
        self._engine = create_engine(self._url, echo=settings.is_development)
        self._session_factory = sessionmaker(bind=self._engine)
    
    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(self._engine)
    
    def get_session(self) -> Session:
        """Get a new session."""
        return self._session_factory()
    
    def __enter__(self) -> Session:
        self._session = self.get_session()
        return self._session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._session.rollback()
        self._session.close()


class QueryHistoryRepository:
    """Repository for query history."""
    
    def __init__(self, session: DatabaseSession = None):
        """
        Initialize repository.
        
        Args:
            session: Database session manager
        """
        self._db = session or DatabaseSession()
    
    def create(
        self,
        query: str,
        response: str,
        reasoning: str = "",
        agents_used: list[str] = None,
        confidence: float = 0.0,
    ) -> QueryHistory:
        """
        Create a new query history entry.
        
        Args:
            query: User query
            response: AI response
            reasoning: Agent reasoning
            agents_used: List of agents used
            confidence: Confidence score
            
        Returns:
            Created QueryHistory
        """
        with self._db as session:
            entry = QueryHistory(
                query=query,
                response=response,
                reasoning=reasoning,
                agents_used=agents_used or [],
                confidence=confidence,
            )
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry
    
    def get_by_id(self, query_id: int) -> Optional[QueryHistory]:
        """Get query by ID."""
        with self._db as session:
            return session.get(QueryHistory, query_id)
    
    def get_all(
        self,
        page: int = 1,
        page_size: int = 10,
    ) -> tuple[list[QueryHistory], int]:
        """
        Get paginated query history.
        
        Args:
            page: Page number (1-indexed)
            page_size: Items per page
            
        Returns:
            Tuple of (items, total_count)
        """
        with self._db as session:
            # Get total count
            total = session.scalar(select(func.count()).select_from(QueryHistory))
            
            # Get paginated items
            offset = (page - 1) * page_size
            items = session.scalars(
                select(QueryHistory)
                .order_by(QueryHistory.created_at.desc())
                .offset(offset)
                .limit(page_size)
            ).all()
            
            return list(items), total
    def get_recent(self, limit: int = 5) -> list[QueryHistory]:
        """
        Get most recent query history items.
        
        Args:
            limit: Number of items to retrieve
            
        Returns:
            List of QueryHistory items
        """
        with self._db as session:
            items = session.scalars(
                select(QueryHistory)
                .order_by(QueryHistory.created_at.desc())
                .limit(limit)
            ).all()
            return list(items)



class FeedbackRepository:
    """Repository for feedback."""
    
    def __init__(self, session: DatabaseSession = None):
        """
        Initialize repository.
        
        Args:
            session: Database session manager
        """
        self._db = session or DatabaseSession()
    
    def create(
        self,
        query_id: int,
        rating: int,
        comment: str = None,
    ) -> Feedback:
        """
        Create a new feedback entry.
        
        Args:
            query_id: Query history ID
            rating: Rating (1-5)
            comment: Optional comment
            
        Returns:
            Created Feedback
        """
        with self._db as session:
            feedback = Feedback(
                query_id=query_id,
                rating=rating,
                comment=comment,
            )
            session.add(feedback)
            session.commit()
            session.refresh(feedback)
            return feedback
    
    def get_by_query_id(self, query_id: int) -> list[Feedback]:
        """Get all feedback for a query."""
        with self._db as session:
            return list(session.scalars(
                select(Feedback).where(Feedback.query_id == query_id)
            ).all())
