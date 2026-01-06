"""Structured logging configuration using structlog."""

import logging
import sys
from typing import Any

import structlog
from structlog.typing import FilteringBoundLogger

from src.config import settings


def configure_logging() -> None:
    """
    Configure structlog for the application.
    
    Log levels:
    - Production (ENVIRONMENT=production): INFO level (only important events)
    - Development (ENVIRONMENT=development): DEBUG level (all events)
    
    Call this once at application startup.
    """
    # Development = DEBUG, Production = INFO
    log_level = logging.DEBUG if settings.is_development else logging.INFO
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    # Shared processors for all loggers
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Development: pretty console output
    # Production: JSON output
    if settings.is_development:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None) -> FilteringBoundLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    
    Usage:
        from src.domain.logger import get_logger
        
        logger = get_logger(__name__)
        
        # Debug logs - shown only in development
        logger.debug("processing_step", step=1, data="...")
        
        # Info logs - shown in production (important events)
        logger.info("ingestion_complete", count=105)
        
        # Warning/Error - always shown
        logger.warning("low_stock", product_id="PROD-001")
        logger.error("api_failed", error=str(e))
    """
    return structlog.get_logger(name)


# Convenience: pre-configured logger for quick imports
logger = get_logger("quanxai")
