from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

from config.config import BaseConfig as Config

# Add base directory to path if needed
if str(Config.BASE_DIR) not in sys.path:
    sys.path.append(str(Config.BASE_DIR))

from contextlib import contextmanager
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    filename=Config.LOG_PATH / 'database.log',
    level=Config.LOG_LEVEL,
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(
            Config.get_database_url(),
            poolclass=QueuePool,
            pool_size=Config.DB_POOL_SIZE,
            max_overflow=Config.DB_MAX_OVERFLOW,
            pool_timeout=Config.DB_POOL_TIMEOUT,
            pool_recycle=Config.DB_POOL_RECYCLE
        )
        self.Session = sessionmaker(bind=self.engine)
        self._session = None

    @property
    def session(self):
        if self._session is None:
            self._session = self.Session()
        return self._session

    def cleanup(self):
        """Cleanup database resources"""
        if self._session is not None:
            try:
                self._session.close()
            except Exception as e:
                logger.error(f"Error closing session: {e}")
            finally:
                self._session = None
        if self.engine:
            try:
                self.engine.dispose()
            except Exception as e:
                logger.error(f"Error disposing engine: {e}")

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()

    def health_check(self):
        """Perform database health check"""
        try:
            # Check basic connectivity
            self.session.execute(text('SELECT 1'))
            
            # Check table statistics
            stats = self.session.execute(text("""
                SELECT 
                    count(*) as total_records,
                    min(datetime) as oldest_record,
                    max(datetime) as newest_record
                FROM container_data
            """)).fetchone()
            
            return {
                'status': 'healthy',
                'total_records': stats[0],
                'oldest_record': stats[1],
                'newest_record': stats[2]
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {'status': 'unhealthy', 'error': str(e)}

    @contextmanager
    def get_session(self):
        """Session context manager"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()

    def bulk_insert(self, records: List[Dict[str, Any]]) -> bool:
        """Improved bulk insert with batching"""
        BATCH_SIZE = 1000
        try:
            with self.get_session() as session:
                for i in range(0, len(records), BATCH_SIZE):
                    batch = records[i:i + BATCH_SIZE]
                    insert_query = text("""
                        INSERT INTO container_data (
                            datetime, ocr_output, camera_id,
                            image_path
                        ) VALUES (
                            :datetime, :ocr_output, :camera_id,
                            :image_path
                        )
                    """)
                    for record in batch:
                        session.execute(insert_query, record)
                    if i == 0 or i + BATCH_SIZE >= len(records):
                        logger.info(f"Inserted records: {i + len(batch)}/{len(records)}")
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            session.rollback()
            raise

    container_results_logger = logging.getLogger('container_results')

    def get_recent_results(self, start_date=None, end_date=None, camera_id=None, search_term=None, limit=None, days=None):
        """Get recent results with query optimization"""
        try:
            # Simple query to get all data
            query = """
                SELECT 
                    id,
                    datetime,
                    ocr_output,
                    camera_id,
                    image_path,
                    track_id,
                    confidence
                FROM container_data
                ORDER BY datetime DESC
            """
            
            self.container_results_logger.debug("Executing simple query to get all records")
            self.container_results_logger.debug(f"Query: {query}")
            
            # Execute query and get results
            results = self.session.execute(text(query))
            rows = results.fetchall()
            
            self.container_results_logger.info(f"Query returned {len(rows)} records")
            
            if rows:
                # Log sample of the data
                sample_size = min(5, len(rows))
                self.container_results_logger.info(f"Sample of first {sample_size} records:")
                for i in range(sample_size):
                    self.container_results_logger.info(
                        f"Record {i+1}: datetime={rows[i][1]}, "
                        f"camera={rows[i][3]}, "
                        f"ocr_output={rows[i][2]}"
                    )

            # Convert the raw rows to object-like dictionaries
            return [
                type('ContainerResult', (), {
                    'id': row[0],
                    'datetime': row[1],
                    'ocr_output': row[2],
                    'camera_id': row[3],
                    'image_path': row[4],
                    'track_id': row[5],
                    'confidence': row[6]
                })
                for row in rows
            ]

        except Exception as e:
            self.container_results_logger.error(f"Error retrieving results: {e}", exc_info=True)
            raise

    def insert_record(self, record: dict) -> None:
        """Insert a new record into the database."""
        try:
            with self.get_session() as session:
                session.execute(text("INSERT INTO container_data (datetime, ocr_output, camera_id, image_path, track_id, confidence) VALUES (:datetime, :ocr_output, :camera_id, :image_path, :track_id, :confidence)"), 
                                {'datetime': record['datetime'], 'ocr_output': record['ocr_output'], 'camera_id': record['camera_id'], 'image_path': record['image_path'], 'track_id': str(record['track_id']), 'confidence': record.get('confidence', None)})
                session.commit()
        except SQLAlchemyError as e:
            logger.error(f"Error inserting record: {e}")
            raise

    def update_record(self, record_id: int, record: dict) -> None:
        """Update an existing record in the database."""
        try:
            with self.get_session() as session:
                session.execute(text("UPDATE container_data SET datetime = :datetime, ocr_output = :ocr_output, camera_id = :camera_id, image_path = :image_path, confidence = :confidence WHERE track_id = :track_id"), 
                                {'datetime': record['datetime'], 'ocr_output': record['ocr_output'], 'camera_id': record['camera_id'], 'image_path': record['image_path'], 'track_id': str(record['track_id']), 'confidence': record.get('confidence', None)})
                session.commit()
        except SQLAlchemyError as e:
            logger.error(f"Error updating record: {e}")
            raise

    def get_record_by_track_id(self, track_id: str) -> Optional[dict]:
        """Retrieve a record by track_id."""
        try:
            with self.get_session() as session:
                # print(f"Looking for track_id: {track_id}")
                result = session.execute(text("SELECT * FROM container_data WHERE TRIM(track_id) = :track_id"), {'track_id': str(track_id)}).fetchone()
                # print(f"Query result for track_id {track_id}: {result}")
                if result:
                    return {
                        'id': result[0],
                        'datetime': result[1],
                        'ocr_output': result[2],
                        'camera_id': result[3],
                        'image_path': result[4],
                        'created_at': result[5],
                        'status': result[6],
                        'track_id': result[7],
                        'confidence': result[8]
                    }
                return None
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving record: {e}")
            raise 

    def get_paginated_results(self, page=1, per_page=20, start_date=None, end_date=None, camera_id=None):
        """Get paginated results with total count"""
        try:
            # Get total records in database
            total_records_query = "SELECT COUNT(*) FROM container_data"
            total_records = self.session.execute(text(total_records_query)).scalar()
            
            # Base query for filtered count
            count_query = """
                SELECT COUNT(DISTINCT track_id)
                FROM container_data
                WHERE 1=1
            """
            
            # Base query for data - using subquery for correct date-based sorting
            query = """
                WITH latest_records AS (
                    SELECT DISTINCT ON (track_id)
                        id,
                        datetime,
                        ocr_output,
                        camera_id,
                        image_path,
                        track_id,
                        confidence
                    FROM container_data
                    WHERE 1=1
            """
            
            params = {}

            # Add filters
            if start_date:
                query += " AND datetime >= :start_date"
                count_query += " AND datetime >= :start_date"
                params['start_date'] = start_date
            if end_date:
                query += " AND datetime <= :end_date"
                count_query += " AND datetime <= :end_date"
                params['end_date'] = end_date
            if camera_id:
                query += " AND camera_id = :camera_id"
                count_query += " AND camera_id = :camera_id"
                params['camera_id'] = camera_id

            # Close the CTE and add final ordering
            query += """
                    ORDER BY track_id, datetime DESC
                )
                SELECT * FROM latest_records
                ORDER BY datetime DESC
                LIMIT :limit OFFSET :offset
            """
            
            # Calculate pagination parameters
            offset = (page - 1) * per_page
            params['limit'] = per_page
            params['offset'] = offset

            # Get filtered count within the same transaction
            filtered_count = self.session.execute(text(count_query), params).scalar()

            # Get paginated results within the same transaction
            results = self.session.execute(text(query), params)
            
            # Convert to objects
            records = [
                type('ContainerResult', (), {
                    'id': row[0],
                    'datetime': row[1],
                    'ocr_output': row[2],
                    'camera_id': row[3],
                    'image_path': row[4],
                    'track_id': row[5],
                    'confidence': row[6]
                })
                for row in results.fetchall()
            ]

            # Calculate pagination metadata
            total_pages = (filtered_count + per_page - 1) // per_page
            has_next = page < total_pages
            has_prev = page > 1

            return {
                'items': records,
                'total': filtered_count,
                'total_records': total_records,  # Total records in database
                'page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'has_next': has_next,
                'has_prev': has_prev
            }

        except Exception as e:
            self.container_results_logger.error(f"Error retrieving paginated results: {e}")
            raise 