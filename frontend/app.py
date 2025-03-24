from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, Response, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from datetime import datetime, timedelta
import pandas as pd
import sys
import logging
import time
from sqlalchemy import text
import psutil
from functools import wraps
from pathlib import Path
from logging.handlers import RotatingFileHandler
from io import BytesIO
import threading
import schedule
import xlsxwriter
import json
from config.exceptions import DatabaseError, ValidationError, StorageError, ConfigurationError
import hashlib
import os
import re
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import weakref
import atexit
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import asyncio
import psycopg2
import uuid
from flask_socketio import SocketIO, emit
from engineio.payload import Payload
from database.db_manager import DatabaseConnectionManager
from broadcast_manager import BroadcastManager

from config.config import BaseConfig as Config

# Set up export-specific logging
export_logger = logging.getLogger('export_operations')
export_logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
log_dir = Path(Config.LOG_PATH)
log_dir.mkdir(parents=True, exist_ok=True)

# Create a file handler for export logs
export_log_file = log_dir / 'export_operations.log'
file_handler = logging.FileHandler(export_log_file)
file_handler.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
export_logger.addHandler(file_handler)

# Initialize application start time
START_TIME = datetime.now()

# Add base directory to path if needed
if str(Config.BASE_DIR) not in sys.path:
    sys.path.append(str(Config.BASE_DIR))

# Now we can import our local modules
from utils.file_utils import get_image_status, cleanup_old_images
from utils.monitoring import REQUEST_DURATION, collect_system_metrics
from database.db_operations import DatabaseManager
from utils.verify_setup import verify_setup

# Initialize Flask app
app = Flask(__name__,
           static_folder=str(Config.BASE_DIR / 'frontend/static'),
           template_folder=str(Config.BASE_DIR / 'frontend/templates'))
app.config.from_object(Config)
app.config['SECRET_KEY'] = 'your-secret-key'  # Change this in production

# Enable CORS with security settings
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Handle proxy headers
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Set up session configuration
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max-content-length

# Initialize caching with Redis if available, fallback to simple
cache_config = {
    'CACHE_TYPE': Config.CACHE_TYPE,
    'CACHE_DEFAULT_TIMEOUT': Config.CACHE_DEFAULT_TIMEOUT
}

if Config.CACHE_TYPE == 'redis':
    cache_config.update({
        'CACHE_REDIS_URL': Config.REDIS_URL,
        'CACHE_KEY_PREFIX': 'ocr_system:'
    })

cache = Cache(app, config=cache_config)

# Initialize rate limiting with more granular controls
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[Config.RATE_LIMIT_DAY, Config.RATE_LIMIT_HOUR],
    storage_uri=Config.REDIS_URL if Config.CACHE_TYPE == 'redis' else None
)

# Initialize logging
log_file = Config.LOG_PATH / 'frontend.log'
handler = RotatingFileHandler(
    log_file, 
    maxBytes=Config.LOG_FILE_MAX_BYTES, 
    backupCount=Config.LOG_FILE_BACKUP_COUNT
)
handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))

# Get logger
logger = logging.getLogger(__name__)
logger.setLevel(Config.LOG_LEVEL)
logger.addHandler(handler)

# Add file handler to werkzeug logger as well
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.addHandler(handler)

# Add SQLAlchemy logging
sqlalchemy_logger = logging.getLogger('sqlalchemy.engine')
sqlalchemy_logger.addHandler(handler)
sqlalchemy_logger.setLevel(logging.INFO)

# Initialize database
db = DatabaseManager()

# Global connection pool with thread safety
db_pool = None
db_pool_lock = threading.Lock()

# Add near the top of the file, after imports
cleanup_event = threading.Event()

def get_db_connection():
    """Get a database connection from the pool"""
    global db_pool
    if db_pool is None:
        with db_pool_lock:
            if db_pool is None:  # Double-check pattern
                db_pool = QueuePool(
                    creator=lambda: psycopg2.connect(Config.get_database_url()),
                    max_overflow=Config.DB_MAX_OVERFLOW,
                    pool_size=Config.DB_POOL_SIZE,
                    timeout=Config.DB_POOL_TIMEOUT,
                    recycle=Config.DB_POOL_RECYCLE
                )
    return db_pool.connect()

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = None
    try:
        conn = get_db_connection()
        yield conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def check_db_connection():
    """Check database connection health"""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

# Initialize Flask-SocketIO with message size configuration
Payload.max_decode_packets = 50
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=20,
    ping_interval=10,
    max_http_buffer_size=1e8,
    logger=True,
    engineio_logger=True,
    path='/socket.io/',
    always_connect=True,
    transports=['websocket', 'polling']
)

# Initialize managers
db_manager = DatabaseConnectionManager()
broadcast_manager = BroadcastManager(socketio)

def init_app(app):
    """Initialize the application"""
    with app.app_context():
        # Initialize database connection
        db_manager.initialize()
        
        # Start broadcast manager
        broadcast_manager.start()
        
        # Start broadcast thread
        broadcast_thread = threading.Thread(target=broadcast_updates, daemon=True)
        broadcast_thread.start()
        
        # Register cleanup
        @app.teardown_appcontext
        def cleanup(exception=None):
            db_manager.cleanup()
            broadcast_manager.stop()

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the evidence_images directory"""
    try:
        # Use the configured path from Config
        image_dir = Config.IMAGE_STORAGE_PATH
        logger.info(f"Serving image {filename} from {image_dir}")
        return send_file(image_dir / filename)
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return "Image not found", 404
    
# def verify_installation():
#     """Verify setup before first request"""
#     if not verify_setup():
#         app.logger.error("Application setup verification failed!")
#         raise SystemExit("Setup verification failed. Run 'python scripts/setup.py' to fix.")

# Add this function for periodic cleanup
def run_periodic_cleanup():
    """Run cleanup tasks in background"""
    def cleanup_task():
        logger.info("Running scheduled cleanup task")
        cleanup_old_images()

    # Run cleanup immediately on startup
    logger.info("Running initial cleanup on startup")
    cleanup_old_images()
    
    # Schedule future cleanups
    schedule.every().day.at("00:00").do(cleanup_task)
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

# Move cleanup function to file_utils.py
def start_cleanup_thread():
    """Start the cleanup thread"""
    cleanup_thread = threading.Thread(target=run_periodic_cleanup, daemon=True)
    cleanup_thread.start()
    logger.info("Cleanup scheduler started")

# Register with app
with app.app_context():
    # verify_installation()
    start_cleanup_thread()  # Start the cleanup scheduler

@app.after_request
def add_security_headers(response):
    """Add security headers to each response"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "font-src 'self' data: https://cdn.jsdelivr.net; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://code.jquery.com https://cdn.socket.io; "
        "connect-src 'self' ws: wss: http: https:;"
    )
    return response

# Add after other metric definitions
SEARCH_REQUESTS = Counter(
    'search_requests_total',
    'Total number of search requests',
    ['status']
)

API_REQUESTS = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'status']
)

CONTAINER_METRICS = Gauge(
    'container_metrics',
    'Container detection metrics',
    ['metric_type']
)

@app.route('/metrics/containers')
@limiter.limit("10 per minute")
def container_metrics():
    """Endpoint for container detection metrics"""
    try:
        # Get data from the last 24 hours
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        results = db.get_recent_results(
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate metrics
        total_detections = len(results)
        unique_containers = len(set(r.ocr_output for r in results))
        camera_counts = {}
        confidence_values = []
        
        for result in results:
            camera_counts[result.camera_id] = camera_counts.get(result.camera_id, 0) + 1
            if hasattr(result, 'confidence') and result.confidence is not None:
                confidence_values.append(result.confidence)
        
        # Update Prometheus metrics
        CONTAINER_METRICS.labels('total_24h').set(total_detections)
        CONTAINER_METRICS.labels('unique_containers_24h').set(unique_containers)
        
        for camera_id, count in camera_counts.items():
            CONTAINER_METRICS.labels(f'camera_{camera_id}_count').set(count)
            
        if confidence_values:
            avg_confidence = sum(confidence_values) / len(confidence_values)
            CONTAINER_METRICS.labels('average_confidence').set(avg_confidence)
        
        # Replace with WebSocket metrics
        CONTAINER_METRICS.labels('active_websocket_connections').set(len(socketio.server.eio.sockets))
        
        metrics_data = {
            'total_detections_24h': total_detections,
            'unique_containers_24h': unique_containers,
            'camera_counts': camera_counts,
            'average_confidence': avg_confidence if confidence_values else None,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(metrics_data)
        
    except Exception as e:
        logger.error(f"Metrics collection error: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

def update_monitoring_metrics(endpoint, status='success'):
    """Update monitoring metrics for endpoints"""
    if endpoint.startswith('/api/'):
        API_REQUESTS.labels(endpoint=endpoint, status=status).inc()
    elif endpoint == '/search':
        SEARCH_REQUESTS.labels(status=status).inc()

# Update the monitor_endpoint decorator
def monitor_endpoint(f):
    """Enhanced decorator to monitor endpoint performance and collect metrics"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            REQUEST_DURATION.labels(endpoint=f.__name__).observe(time.time() - start_time)
            update_monitoring_metrics(request.path)
            return result
        except Exception as e:
            logger.error(f"Endpoint {f.__name__} failed: {e}")
            update_monitoring_metrics(request.path, 'error')
            raise
    return wrapper

def show_message(message, type="info"):
    """Show message in dashboard with all required template variables"""
    return render_template(
        'dashboard.html',
        results=[],  # Empty results list
        message=message,
        message_type=type,
        metrics=collect_system_metrics(),
        start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d'),
        camera_id=None
    )

def cache_key_with_params():
    """Generate cache key based on request parameters"""
    args = request.args.copy()
    args.pop('_', None)  # Remove any cache busting parameters
    key_parts = [
        request.path,
        str(sorted(args.items()))
    ]
    return 'view/%s' % hashlib.md5(str(key_parts).encode()).hexdigest()

@app.route('/')
@limiter.limit("30 per minute")  # More permissive for dashboard
@monitor_endpoint
def index():
    """Main dashboard route with proper error handling and date filtering"""
    try:
        # Get filter parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        camera_id = request.args.get('camera_id')
        date_filtering = request.args.get('dateFiltering', type=bool, default=False)
        
        # Get pagination parameters
        page = request.args.get('page', type=int, default=1)
        per_page = request.args.get('per_page', type=int, default=20)

        logger.info("Index parameters: start_date=%s, end_date=%s, camera_id=%s, date_filtering=%s, page=%s", 
                   start_date, end_date, camera_id, date_filtering, page)

        # Process dates if date filtering is enabled
        if date_filtering and (start_date or end_date):
            try:
                if start_date:
                    start_date = datetime.strptime(start_date, '%Y-%m-%d').replace(hour=0, minute=0, second=0)
                if end_date:
                    # Add one day to end_date to include the entire day
                    end_date = (datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59) 
                              + timedelta(days=1))
            except ValueError as e:
                logger.error(f"Date parsing error: {e}")
                flash("Invalid date format. Please use YYYY-MM-DD format.", "error")
                start_date = None
                end_date = None
        else:
            # Default to last 24 hours if no date filtering
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)

        # Get paginated data from database
        try:
            paginated_results = db.get_paginated_results(
                page=page,
                per_page=per_page,
                start_date=start_date,
                end_date=end_date,
                camera_id=camera_id if camera_id else None
            )
            
            logger.info(f"Retrieved {len(paginated_results['items'])} records for page {page}")
        except DatabaseError as e:
            logger.error(f"Database query error: {e}")
            raise

        # Collect system metrics
        try:
            system_metrics = collect_system_metrics()
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            system_metrics = None

        # Format dates for template
        template_start_date = start_date.strftime('%Y-%m-%d') if start_date else None
        template_end_date = (end_date - timedelta(days=1)).strftime('%Y-%m-%d') if end_date else None

        return render_template(
            'dashboard.html',
            results=paginated_results['items'],
            pagination=paginated_results,
            metrics=system_metrics,
            start_date=template_start_date,
            end_date=template_end_date,
            camera_id=camera_id if camera_id else '',
            date_filtering=date_filtering,
            last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

    except DatabaseError as e:
        flash(f"Database error: {str(e)}", "error")
        return render_template('error.html', 
                             error="Database Error", 
                             details=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in index route: {e}", exc_info=True)
        return render_template('error.html', 
                             error="An unexpected error occurred", 
                             details=str(e))

@app.route('/export')
def export_data():
    """Export data to Excel"""
    try:
        # Get filter parameters
        days = request.args.get('days', default=30, type=int)
        camera_id = request.args.get('camera_id', default=None, type=str)
        
        # Log the received parameters
        logger.info("Exporting data with parameters: days=%d, camera_id=%s", days, camera_id)
        
        # Get data from database
        results = db.get_recent_results(days=days, camera_id=camera_id)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Create Excel file
        excel_file = Config.LOG_PATH / f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        df.to_excel(excel_file, index=False)
        
        return send_file(excel_file, as_attachment=True)
    except Exception as e:
        logger.error(f"Error in export: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/search')
@limiter.limit("20 per minute")
@cache.memoize(timeout=60)
def search():
    """Enhanced search functionality with proper validation and error handling"""
    try:
        container_number = request.args.get('container', '').strip()
        
        # Validate input
        if not container_number:
            return jsonify({
                'error': 'No container number provided',
                'status': 'error'
            }), 400
            
        # Basic container number validation (customize based on your format)
        if not re.match(r'^[A-Z]{4}\d{7}$', container_number):
            return jsonify({
                'error': 'Invalid container number format. Expected format: ABCD1234567',
                'status': 'error'
            }), 400
            
        logger.info("Searching for container number: %s", container_number)
        
        # Use get_recent_results with search parameter
        try:
            results = db.get_recent_results(
                search_term=container_number,
                limit=50  # Limit the number of results
            )
            
            # Format results for JSON response
            formatted_results = [{
                'datetime': result.datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'ocr_output': result.ocr_output,
                'camera_id': result.camera_id,
                'confidence': getattr(result, 'confidence', None),
                'image_path': result.image_path,
                'image_available': get_image_status(result.image_path)[0]
            } for result in results]
            
            return jsonify({
                'status': 'success',
                'count': len(formatted_results),
                'results': formatted_results
            })
            
        except DatabaseError as e:
            logger.error(f"Database error during search: {e}")
            raise
            
    except DatabaseError as e:
        return jsonify({
            'error': f'Database error: {str(e)}',
            'status': 'error'
        }), 500
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred',
            'status': 'error'
        }), 500

@app.route('/health')
@limiter.limit("60 per minute")
@cache.cached(timeout=10, key_prefix='health')
def health_check():
    """Enhanced health check endpoint with comprehensive system checks"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }

        # Database health
        try:
            db_health = db.health_check()
            health_status['components']['database'] = {
                'status': 'healthy',
                'details': db_health
            }
        except Exception as e:
            health_status['components']['database'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'degraded'

        # File system health
        try:
            disk = psutil.disk_usage('/')
            health_status['components']['disk'] = {
                'status': 'healthy' if disk.percent < 90 else 'warning',
                'usage_percent': disk.percent,
                'free_space': f"{disk.free / (1024**3):.2f}GB",
                'total_space': f"{disk.total / (1024**3):.2f}GB"
            }
            if disk.percent >= 90:
                health_status['status'] = 'degraded'
        except Exception as e:
            health_status['components']['disk'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'degraded'

        # Memory usage
        try:
            memory = psutil.virtual_memory()
            health_status['components']['memory'] = {
                'status': 'healthy' if memory.percent < 90 else 'warning',
                'usage_percent': memory.percent,
                'available': f"{memory.available / (1024**3):.2f}GB",
                'total': f"{memory.total / (1024**3):.2f}GB"
            }
            if memory.percent >= 90:
                health_status['status'] = 'degraded'
        except Exception as e:
            health_status['components']['memory'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'degraded'

        # Cache health
        try:
            if Config.CACHE_TYPE == 'redis':
                redis_client = cache.get_redis()
                redis_info = redis_client.info()
                health_status['components']['cache'] = {
                    'status': 'healthy',
                    'type': 'redis',
                    'version': redis_info.get('redis_version'),
                    'used_memory': f"{redis_info.get('used_memory_human', 'N/A')}"
                }
            else:
                health_status['components']['cache'] = {
                    'status': 'healthy',
                    'type': Config.CACHE_TYPE
                }
        except Exception as e:
            health_status['components']['cache'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'degraded'

        # Evidence directory check
        try:
            evidence_dir = Config.IMAGE_STORAGE_PATH
            if not evidence_dir.exists():
                raise StorageError("Evidence directory does not exist")
            if not os.access(evidence_dir, os.W_OK):
                raise StorageError("Evidence directory is not writable")
            health_status['components']['storage'] = {
                'status': 'healthy',
                'path': str(evidence_dir)
            }
        except Exception as e:
            health_status['components']['storage'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'degraded'

        # System uptime and load
        try:
            load1, load5, load15 = psutil.getloadavg()
            cpu_count = psutil.cpu_count()
            health_status['components']['system'] = {
                'status': 'healthy',
                'uptime': str(datetime.now() - START_TIME),
                'load_averages': {
                    '1min': load1,
                    '5min': load5,
                    '15min': load15
                },
                'cpu_count': cpu_count
            }
            # Mark as warning if load is high
            if load5 > cpu_count:
                health_status['components']['system']['status'] = 'warning'
                health_status['status'] = 'degraded'
        except Exception as e:
            health_status['components']['system'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'degraded'

        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(health_status), status_code

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/docs')
def api_docs():
    """
    API Documentation
    ---
    responses:
      200:
        description: API documentation
    """
    return render_template('api_docs.html')

@app.route('/api/v1/containers', methods=['GET'])
@limiter.limit("30 per minute")
@cache.memoize(timeout=60)
def api_get_containers():
    """
    API endpoint to get container detections with filtering
    ---
    parameters:
      - name: start_date
        in: query
        type: string
        description: Start date (YYYY-MM-DD)
      - name: end_date
        in: query
        type: string
        description: End date (YYYY-MM-DD)
      - name: camera_id
        in: query
        type: string
        description: Camera ID filter
      - name: confidence
        in: query
        type: float
        description: Minimum confidence threshold
      - name: limit
        in: query
        type: integer
        description: Maximum number of results
    """
    try:
        # Get and validate parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        camera_id = request.args.get('camera_id')
        confidence = request.args.get('confidence', type=float)
        limit = request.args.get('limit', type=int, default=100)
        
        # Convert dates if provided
        try:
            if start_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        except ValueError:
            return jsonify({
                'error': 'Invalid date format. Use YYYY-MM-DD',
                'status': 'error'
            }), 400
            
        # Get results
        results = db.get_recent_results(
            start_date=start_date,
            end_date=end_date,
            camera_id=camera_id,
            limit=limit
        )
        
        # Filter by confidence if specified
        if confidence is not None:
            results = [r for r in results if getattr(r, 'confidence', 0) >= confidence]
        
        # Format results
        formatted_results = [{
            'id': getattr(result, 'id', None),
            'datetime': result.datetime.isoformat(),
            'ocr_output': result.ocr_output,
            'camera_id': result.camera_id,
            'confidence': getattr(result, 'confidence', None),
            'image_path': result.image_path,
            'image_available': get_image_status(result.image_path)[0]
        } for result in results]
        
        return jsonify({
            'status': 'success',
            'count': len(formatted_results),
            'results': formatted_results
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/v1/container/<container_id>')
@limiter.limit("30 per minute")
@cache.memoize(timeout=60)
def get_container(container_id):
    """Get detailed information for a specific container"""
    try:
        # Validate container ID format
        if not re.match(r'^[A-Z]{4}\d{7}$', container_id):
            return jsonify({
                'error': 'Invalid container number format',
                'status': 'error'
            }), 400
            
        # Get all detections for this container
        results = db.get_recent_results(search_term=container_id)
        
        if not results:
            return jsonify({
                'error': 'Container not found',
                'status': 'error'
            }), 404
            
        # Format detections
        detections = [{
            'datetime': result.datetime.isoformat(),
            'camera_id': result.camera_id,
            'confidence': getattr(result, 'confidence', None),
            'image_path': result.image_path,
            'image_available': get_image_status(result.image_path)[0]
        } for result in results]
        
        # Calculate statistics
        confidences = [d['confidence'] for d in detections if d['confidence'] is not None]
        stats = {
            'total_detections': len(detections),
            'unique_cameras': len(set(d['camera_id'] for d in detections)),
            'avg_confidence': sum(confidences) / len(confidences) if confidences else None,
            'first_seen': min(r.datetime for r in results).isoformat(),
            'last_seen': max(r.datetime for r in results).isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'container_id': container_id,
            'statistics': stats,
            'detections': detections
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.context_processor
def utility_processor():
    """Add utility functions to template context"""
    def get_current_year():
        return datetime.now().year
    
    from utils.file_utils import get_image_status
    return dict(
        current_year=get_current_year(),
        get_image_status=get_image_status,
        min=min  # Add built-in min function to template context
    )

@app.route('/export_excel')
@limiter.limit("10 per minute")
@monitor_endpoint
def export_excel():
    try:
        export_logger.info("Starting Excel export process")
        export_logger.debug(f"Request parameters: {request.args}")
        export_logger.debug(f"Request headers: {dict(request.headers)}")
        
        # Get parameters from request
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        camera_id = request.args.get('camera_id')

        export_logger.info(f"Export parameters - Start Date: {start_date}, End Date: {end_date}, Camera ID: {camera_id}")

        # Convert dates to datetime objects
        if start_date:
            try:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                start_date = start_date.replace(hour=0, minute=0, second=0)
                export_logger.debug(f"Parsed start_date: {start_date}")
            except ValueError as e:
                export_logger.error(f"Error parsing start_date: {e}")
                return jsonify({"error": "Invalid start date format"}), 400
        else:
            start_date = datetime.now() - timedelta(days=30)
            export_logger.info(f"No start_date provided, using default: {start_date}")

        if end_date:
            try:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                end_date = end_date.replace(hour=23, minute=59, second=59)
                export_logger.debug(f"Parsed end_date: {end_date}")
            except ValueError as e:
                export_logger.error(f"Error parsing end_date: {e}")
                return jsonify({"error": "Invalid end date format"}), 400
        else:
            end_date = datetime.now()
            export_logger.info(f"No end_date provided, using default: {end_date}")

        # Log the time range being queried
        time_range = end_date - start_date
        export_logger.info(f"Querying data for time range: {time_range.days} days, {time_range.seconds // 3600} hours")
        
        # Get database connection
        export_logger.debug("Getting database connection")
        db = DatabaseManager()

        # Verify database connection
        try:
            db.session.execute(text("SELECT 1"))
            export_logger.debug("Database connection verified")
        except Exception as e:
            export_logger.error(f"Database connection failed: {e}", exc_info=True)
            return jsonify({"error": "Database connection failed"}), 500

        # Get results from database
        export_logger.info("Fetching results from database")
        try:
            results = db.get_recent_results(
                start_date=start_date,
                end_date=end_date,
                camera_id=camera_id if camera_id else None
            )
            export_logger.info(f"Retrieved {len(results) if results else 0} records from database")
            
            # Log sample of results if available
            if results:
                sample_size = min(3, len(results))
                export_logger.debug(f"Sample of first {sample_size} results:")
                for i, result in enumerate(results[:sample_size]):
                    export_logger.debug(f"Result {i+1}: datetime={result.datetime}, camera={result.camera_id}, track_id={result.track_id}")
            
        except Exception as e:
            export_logger.error(f"Database query failed: {e}", exc_info=True)
            return jsonify({"error": "Failed to retrieve data from database"}), 500

        if not results:
            export_logger.warning("No data found for the specified filters")
            export_logger.debug(f"Query parameters: start_date={start_date}, end_date={end_date}, camera_id={camera_id}")
            
            # Get database statistics
            try:
                stats = db.session.execute(text("""
                    SELECT 
                        COUNT(*) as total_records,
                        MIN(datetime) as earliest_record,
                        MAX(datetime) as latest_record,
                        COUNT(DISTINCT camera_id) as unique_cameras
                    FROM container_data
                """)).fetchone()
                
                export_logger.warning(
                    "Database statistics:\n"
                    f"Total records: {stats[0]}\n"
                    f"Earliest record: {stats[1]}\n"
                    f"Latest record: {stats[2]}\n"
                    f"Unique cameras: {stats[3]}"
                )

                # Create a more informative error message
                error_message = "No data found for the selected filters. "
                if stats[0] > 0:  # If there are records in the database
                    error_message += f"Available data range is from {stats[1].strftime('%Y-%m-%d %H:%M')} to {stats[2].strftime('%Y-%m-%d %H:%M')}. "
                    if stats[3] == 1:
                        error_message += f"Data is available for {stats[3]} camera. "
                    else:
                        error_message += f"Data is available for {stats[3]} cameras. "
                    error_message += "Please adjust your date range accordingly."
                
                return jsonify({"error": error_message}), 404
                
            except Exception as e:
                export_logger.error(f"Failed to get database statistics: {e}")
                return jsonify({"error": "No data found for the selected filters"}), 404

        # Create DataFrame
        export_logger.debug("Creating DataFrame from results")
        data = []
        for result in results:
            row = {
                'Date/Time': result.datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'Container Number': result.ocr_output,
                'Camera': result.camera_id
            }
            data.append(row)

        df = pd.DataFrame(data)
        export_logger.debug(f"DataFrame created with {len(df)} rows and columns: {list(df.columns)}")

        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"container_data_{timestamp}.xlsx"
        export_path = os.path.join(Config.TEMP_DIR, filename)
        export_logger.info(f"Generating Excel file: {filename}")

        # Create Excel file with enhanced formatting
        try:
            with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Container Data')
                
                # Get the workbook and the worksheet
                workbook = writer.book
                worksheet = writer.sheets['Container Data']

                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column = [cell for cell in column]
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = (max_length + 2)
                    worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

                export_logger.debug("Excel file formatted and written successfully")
        except Exception as e:
            export_logger.error(f"Error creating Excel file: {e}", exc_info=True)
            return jsonify({"error": "Error generating Excel file"}), 500

        # Verify file was created
        if not os.path.exists(export_path):
            export_logger.error(f"Excel file was not created at {export_path}")
            return jsonify({"error": "Failed to create Excel file"}), 500

        export_logger.debug(f"File size: {os.path.getsize(export_path)} bytes")

        # Send file
        export_logger.info("Sending Excel file to client")
        return send_file(
            export_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        export_logger.error(f"Unexpected error during export: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# 1. Database Connection Handling
@app.before_request
def check_db_connection():
    """Check database connection before each request"""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                if cur.fetchone()[0] != 1:
                    raise Exception("Database health check failed")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return jsonify({'error': 'Service temporarily unavailable'}), 503

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('system_status', {
        'status': 'connected',
        'timestamp': datetime.now().isoformat()
    })
    # Send initial system metrics
    send_system_metrics()

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

def broadcast_updates():
    """Background task to broadcast updates to all connected clients"""
    last_processed_records = {}  # Keep track of last processed records by track_id
    
    while not cleanup_event.is_set():
        try:
            with get_db() as conn:
                with conn.cursor() as cursor:
                    # Get latest records including updates
                    cursor.execute("""
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
                    """)
                    
                    results = cursor.fetchall()
                    if results:
                        new_records = []
                        updated_records = []
                        
                        for row in results:
                            record = {
                                'id': row[0],
                                'datetime': row[1].isoformat() if row[1] else None,
                                'ocr_output': row[2],
                                'camera_id': row[3],
                                'image_path': row[4],
                                'track_id': str(row[5]),  # Convert track_id to string
                                'confidence': float(row[6]) if row[6] is not None else None,
                                'created_at': row[7].isoformat() if row[7] else None,
                                'image_available': get_image_status(row[4])[0]
                            }
                            
                            track_id = str(row[5])
                            
                            if track_id in last_processed_records:
                                old_record = last_processed_records[track_id]
                                if (record['confidence'] != old_record['confidence'] or 
                                    record['ocr_output'] != old_record['ocr_output'] or
                                    record['image_path'] != old_record['image_path'] or
                                    record['datetime'] != old_record['datetime']):
                                    updated_records.append(record)
                                    logger.info(f"Record updated - Track ID: {track_id}, "
                                              f"Old confidence: {old_record['confidence']}, "
                                              f"New confidence: {record['confidence']}")
                            else:
                                new_records.append(record)
                                logger.info(f"New record - Track ID: {track_id}, "
                                          f"Confidence: {record['confidence']}")
                            
                            last_processed_records[track_id] = record

                        # Get current page info
                        paginated_results = db.get_paginated_results(page=1, per_page=20)
                        
                        if new_records:
                            logger.info(f"Broadcasting {len(new_records)} new records")
                            socketio.emit('new_records', {
                                'records': new_records,
                                'pagination': {
                                    'total': paginated_results['total'],
                                    'page': 1,
                                    'per_page': 20,
                                    'total_pages': paginated_results['total_pages']
                                }
                            }, namespace='/')
                        
                        if updated_records:
                            logger.info(f"Broadcasting {len(updated_records)} updated records")
                            socketio.emit('updated_records', {
                                'records': updated_records,
                                'pagination': {
                                    'total': paginated_results['total'],
                                    'page': 1,
                                    'per_page': 20,
                                    'total_pages': paginated_results['total_pages']
                                }
                            }, namespace='/')
                        
        except Exception as e:
            logger.error(f"Error in broadcast_updates: {e}", exc_info=True)
        
        # Use cleanup_event for controlled sleep
        cleanup_event.wait(1.0)  # Check for updates every second

# Start background task for broadcasting updates
@socketio.on_error_default
def default_error_handler(e):
    """Handle WebSocket errors"""
    logger.error(f"WebSocket error: {e}")
    emit('error', {'error': str(e)})

# Shutdown handler
def shutdown_cleanup():
    """Clean up resources on application shutdown"""
    logger.info("Starting shutdown cleanup...")
    
    # Signal threads to stop
    cleanup_event.set()
    
    # Wait for cleanup thread to finish
    if 'cleanup_thread' in globals() and cleanup_thread:
        cleanup_thread.join(timeout=5)
        logger.info("Cleanup thread joined")
    
    # Wait for broadcast thread to finish
    if 'broadcast_thread' in globals() and broadcast_thread:
        broadcast_thread.join(timeout=5)
        logger.info("Broadcast thread joined")
    
    # Clean up database connections
    try:
        if hasattr(db, 'engine'):
            db.engine.dispose()
            logger.info("Database connections disposed")
    except Exception as e:
        logger.error(f"Error disposing database connections: {e}")

# Register shutdown handler
atexit.register(shutdown_cleanup)

# Add cache invalidation endpoint
@app.route('/admin/cache/invalidate/<pattern>', methods=['POST'])
@limiter.limit("30 per minute")
def invalidate_cache_pattern(pattern):
    """Invalidate cache entries matching a pattern"""
    try:
        invalidate_cache_for(pattern)
        return jsonify({'message': f'Cache invalidated for pattern: {pattern}'}), 200
    except Exception as e:
        logger.error(f"Cache invalidation failed: {e}")
        return jsonify({'error': 'Failed to invalidate cache'}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', 
                         error="Page not found", 
                         details="The requested page could not be found."), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}", exc_info=True)
    return render_template('error.html', 
                         error="Internal server error", 
                         details="An unexpected error occurred."), 500

@app.errorhandler(DatabaseError)
def handle_database_error(error):
    logger.error(f"Database error: {error}", exc_info=True)
    return render_template('error.html', 
                         error="Database error", 
                         details=str(error)), 500

@app.errorhandler(ValidationError)
def handle_validation_error(error):
    return render_template('error.html', 
                         error="Validation error", 
                         details=str(error)), 400

@app.errorhandler(ConfigurationError)
def handle_config_error(error):
    logger.error(f"Configuration error: {error}", exc_info=True)
    return render_template('error.html', 
                         error="Configuration error", 
                         details=str(error)), 500

# Add cache management endpoints for admin use
@app.route('/admin/cache/clear', methods=['POST'])
@limiter.limit("5 per hour")  # Strict limit for cache management
def clear_cache():
    try:
        with app.app_context():
            cache.clear()
        return jsonify({'message': 'Cache cleared successfully'}), 200
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        return jsonify({'error': 'Failed to clear cache'}), 500

# Add function to invalidate specific cache entries
def invalidate_cache_for(key_pattern):
    """Invalidate cache entries matching a pattern"""
    try:
        if Config.CACHE_TYPE == 'redis':
            redis_client = cache.get_redis()
            keys = redis_client.keys(f"*{key_pattern}*")
            if keys:
                redis_client.delete(*keys)
        else:
            cache.clear()
    except Exception as e:
        logger.error(f"Cache invalidation failed for pattern {key_pattern}: {e}")

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    app.logger.error(f"Unhandled error: {error}", exc_info=True)
    return jsonify({
        'error': 'Internal server error',
        'message': str(error) if app.debug else 'An unexpected error occurred'
    }), 500

def get_system_metrics():
    """Collect system metrics"""
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Get camera status (implement your actual camera status check here)
        camera_status = [
            {'id': f'CAM_{i}', 'status': True} for i in range(1, 6)  # Example status
        ]
        
        return {
            'cpu_usage': round(cpu_usage, 2),
            'memory_usage': round(memory_usage, 2),
            'camera_status': camera_status,
            'uptime': (datetime.now() - START_TIME).total_seconds()
        }
    except Exception as e:
        logger.error(f"Error collecting system metrics: {e}")
        return None

def send_system_metrics():
    """Send system metrics to connected clients"""
    metrics = get_system_metrics()
    if metrics:
        socketio.emit('system_metrics', metrics)

# Schedule periodic system metrics updates
def schedule_metrics_update():
    """Schedule periodic system metrics updates"""
    while True:
        send_system_metrics()
        time.sleep(5)  # Update every 5 seconds

# Remove the @app.before_first_request decorator and create a new function
def init_metrics_thread():
    """Initialize metrics thread and other startup tasks"""
    # Start the metrics update thread
    thread = threading.Thread(target=schedule_metrics_update)
    thread.daemon = True
    thread.start()
    logger.info("Metrics update thread started")

if __name__ == '__main__':
    try:
        init_app(app)
        # Initialize metrics thread before running the app
        init_metrics_thread()
        socketio.run(
            app,
            host='0.0.0.0',
            port=8014,
            debug=False,
            allow_unsafe_werkzeug=True,
            use_reloader=False,
            log_output=True
        )
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
    finally:
        logger.info("Running shutdown cleanup...")
        shutdown_cleanup()
        logger.info("Shutdown complete") 