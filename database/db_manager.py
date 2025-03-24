from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
import threading
import time
import logging
from prometheus_client import Counter, Histogram, Gauge
import psycopg2
from contextlib import contextmanager
from config.config import BaseConfig as Config

# Metrics
DB_POOL_SIZE = Gauge('db_pool_size', 'Current database pool size')
DB_POOL_AVAILABLE = Gauge('db_pool_available', 'Available connections in pool')
DB_QUERY_DURATION = Histogram('db_query_duration_seconds', 'Database query duration')
DB_ERRORS = Counter('db_errors_total', 'Database error count')

logger = logging.getLogger(__name__)

class DatabaseConnectionManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseConnectionManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.max_retries = 3
        self.retry_delay = 1
        self.engine = None
        self.Session = None
        self._lock = threading.Lock()
        self._initialized = True

    def initialize(self):
        """Initialize database connection"""
        if self.engine is None:
            with self._lock:
                if self.engine is None:
                    self.engine = create_engine(
                        Config.get_database_url(),
                        poolclass=QueuePool,
                        pool_size=Config.DB_POOL_SIZE,
                        max_overflow=Config.DB_MAX_OVERFLOW,
                        pool_timeout=Config.DB_POOL_TIMEOUT,
                        pool_recycle=Config.DB_POOL_RECYCLE
                    )
                    self.Session = scoped_session(sessionmaker(bind=self.engine))
                    logger.info("Database connection initialized")

    def cleanup(self):
        """Cleanup database resources"""
        if self.Session:
            self.Session.remove()
        if self.engine:
            self.engine.dispose()
        logger.info("Database resources cleaned up")

    @contextmanager
    def get_session(self):
        """Get a database session"""
        if not self.Session:
            self.initialize()
        
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            DB_ERRORS.inc()
            raise
        finally:
            session.close()

    def execute_query(self, query, params=None, timeout=10):
        """Execute a query with retry logic"""
        for attempt in range(self.max_retries):
            try:
                with DB_QUERY_DURATION.time():
                    with self.get_session() as session:
                        if timeout:
                            session.execute(text(f"SET statement_timeout = {timeout * 1000}"))
                        result = session.execute(text(query), params or {})
                        return result.fetchall()
            except Exception as e:
                logger.error(f"Query attempt {attempt + 1} failed: {e}")
                DB_ERRORS.inc()
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay * (attempt + 1))

    def execute_write(self, query, params=None):
        """Execute a write query"""
        with self.get_session() as session:
            session.execute(text(query), params or {})

db_manager = DatabaseConnectionManager() 