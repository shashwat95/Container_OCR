import threading
import time
import logging
from datetime import datetime
from cachetools import LRUCache
from prometheus_client import Counter, Histogram
from database.db_manager import db_manager
from utils.file_utils import get_image_status

# Metrics
BROADCAST_DURATION = Histogram('broadcast_duration_seconds', 'Broadcast operation duration')
BROADCAST_ERRORS = Counter('broadcast_errors_total', 'Broadcast error count')
RECORDS_PROCESSED = Counter('records_processed_total', 'Total records processed')
NEW_RECORDS_SENT = Counter('new_records_sent_total', 'New records sent')
UPDATED_RECORDS_SENT = Counter('updated_records_sent_total', 'Updated records sent')

logger = logging.getLogger(__name__)

class BroadcastManager:
    def __init__(self, socketio, batch_size=100, rate_limit=1.0, cache_size=1000):
        self.socketio = socketio
        self.batch_size = batch_size
        self.rate_limit = rate_limit
        self.last_broadcast = 0
        self.last_processed_records = LRUCache(maxsize=cache_size)
        self._lock = threading.Lock()
        self.cleanup_event = threading.Event()

    def start(self):
        """Start the broadcast thread"""
        self.broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        self.broadcast_thread.start()
        logger.info("Broadcast thread started")

    def stop(self):
        """Stop the broadcast thread"""
        self.cleanup_event.set()
        if hasattr(self, 'broadcast_thread'):
            self.broadcast_thread.join(timeout=5)
        logger.info("Broadcast thread stopped")

    def _broadcast_loop(self):
        """Main broadcast loop"""
        while not self.cleanup_event.is_set():
            try:
                with BROADCAST_DURATION.time():
                    self._process_updates()
                
                # Rate limiting
                current_time = time.time()
                if current_time - self.last_broadcast < self.rate_limit:
                    time.sleep(self.rate_limit - (current_time - self.last_broadcast))
                
            except Exception as e:
                BROADCAST_ERRORS.inc()
                logger.error(f"Broadcast error: {e}", exc_info=True)
                time.sleep(1)

    def _process_updates(self):
        """Process and broadcast updates"""
        records = self._fetch_records()
        if not records:
            return

        RECORDS_PROCESSED.inc(len(records))

        for batch in self._batch_records(records):
            new_records, updated_records = self._process_batch(batch)
            self._emit_updates(new_records, updated_records)

        self.last_broadcast = time.time()

    def _fetch_records(self):
        """Fetch recent records from database"""
        query = """
            WITH latest_records AS (
                SELECT DISTINCT ON (track_id)
                    id, datetime, ocr_output, camera_id, image_path, 
                    track_id, confidence, created_at
                FROM container_data
                WHERE datetime > NOW() - INTERVAL '5 minute'
                ORDER BY track_id, confidence DESC, datetime DESC
            )
            SELECT * FROM latest_records
            ORDER BY datetime DESC
        """
        return db_manager.execute_query(query, timeout=10)

    def _batch_records(self, records):
        """Split records into batches"""
        for i in range(0, len(records), self.batch_size):
            yield records[i:i + self.batch_size]

    def _process_batch(self, batch):
        """Process a batch of records"""
        new_records = []
        updated_records = []

        for row in batch:
            record = self._format_record(row)
            track_id = str(row[5])

            with self._lock:
                if track_id in self.last_processed_records:
                    if self._record_changed(record, self.last_processed_records[track_id]):
                        updated_records.append(record)
                        UPDATED_RECORDS_SENT.inc()
                else:
                    new_records.append(record)
                    NEW_RECORDS_SENT.inc()
                self.last_processed_records[track_id] = record

        return new_records, updated_records

    def _format_record(self, row):
        """Format database record for transmission"""
        return {
            'id': row[0],
            'datetime': row[1].isoformat() if row[1] else None,
            'ocr_output': row[2],
            'camera_id': row[3],
            'image_path': row[4],
            'track_id': str(row[5]),
            'confidence': float(row[6]) if row[6] is not None else None,
            'created_at': row[7].isoformat() if row[7] else None,
            'image_available': get_image_status(row[4])[0]
        }

    def _record_changed(self, new_record, old_record):
        """Check if record has changed"""
        return (
            new_record['confidence'] != old_record['confidence'] or
            new_record['ocr_output'] != old_record['ocr_output'] or
            new_record['image_path'] != old_record['image_path'] or
            new_record['datetime'] != old_record['datetime']
        )

    def _emit_updates(self, new_records, updated_records):
        """Emit updates via WebSocket"""
        if new_records:
            logger.info(f"Broadcasting {len(new_records)} new records")
            self.socketio.emit('new_records', {
                'records': new_records,
                'pagination': self._get_pagination_info()
            }, namespace='/')

        if updated_records:
            logger.info(f"Broadcasting {len(updated_records)} updated records")
            self.socketio.emit('updated_records', {
                'records': updated_records,
                'pagination': self._get_pagination_info()
            }, namespace='/')

    def _get_pagination_info(self):
        """Get current pagination information"""
        query = "SELECT COUNT(*) FROM container_data"
        total_count = db_manager.execute_query(query)[0][0]
        return {
            'total': total_count,
            'page': 1,
            'per_page': 20,
            'total_pages': (total_count + 19) // 20
        } 