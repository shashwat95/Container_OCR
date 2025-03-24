// Main JavaScript file
"use strict";

// Global configuration
window.CONFIG = {
    maxRows: 100,
    heartbeatInterval: 35000,
    reconnectDelay: 3000,
    maxReconnectAttempts: 5
};

// Global state management
window.state = {
    socket: null,
    heartbeatTimeout: null,
    reconnectAttempts: 0,
    connectionState: 'disconnected',
    resources: {
        intervals: new Set(),
        timeouts: new Set()
    }
};

// Add this at the top level of the file
const trackIdMap = new Map();

// Add to the top level, after CONFIG
const ERROR_MESSAGES = {
    CONNECTION_LOST: 'Connection to server lost. Attempting to reconnect...',
    MAX_RETRIES: 'Connection lost. Please refresh the page.',
    PARSE_ERROR: 'Error processing server data.',
    SERVER_ERROR: 'Server error occurred.',
    RECOVERY_SUCCESS: 'Connection restored successfully.'
};

// Add performance utilities
const utils = {
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    createDocumentFragment() {
        return document.createDocumentFragment();
    },

    batchDOMOperations(operations) {
        return new Promise(resolve => {
            requestAnimationFrame(() => {
                const fragment = utils.createDocumentFragment();
                operations.forEach(op => op(fragment));
                resolve(fragment);
            });
        });
    }
};

// Resource management functions
function addInterval(callback, delay) {
    const id = setInterval(callback, delay);
    state.resources.intervals.add(id);
    return id;
}

function addTimeout(callback, delay) {
    const id = setTimeout(() => {
        state.resources.timeouts.delete(id);
        callback();
    }, delay);
    state.resources.timeouts.add(id);
    return id;
}

function cleanup() {
    // Clear all intervals
    state.resources.intervals.forEach(id => {
        clearInterval(id);
    });
    state.resources.intervals.clear();

    // Clear all timeouts
    state.resources.timeouts.forEach(id => {
        clearTimeout(id);
    });
    state.resources.timeouts.clear();

    // Disconnect WebSocket
    if (state.socket) {
        state.socket.disconnect();
        state.socket = null;
    }

    // Clear heartbeat timeout
    if (state.heartbeatTimeout) {
        clearTimeout(state.heartbeatTimeout);
        state.heartbeatTimeout = null;
    }
}

// UI Functions
function showLoading() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) loadingOverlay.style.display = 'none';
}

function showError(message) {
    const errorDiv = document.getElementById('error-message');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    }
}

// Event handlers setup
function setupEventHandlers() {
    // Handle visibility changes
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            cleanup();
        } else {
            initializeWebSocket();
        }
    });

    // Handle page unload
    window.addEventListener('beforeunload', cleanup);
}

// Initialize UI components
function initializeUI() {
    console.log('Initializing UI components...');
    setupEventHandlers();
}

// System uptime counter
function initializeUptimeCounter() {
    const uptimeElement = document.getElementById('system-uptime');
    if (!uptimeElement) return;

    // Check if we have uptime seconds data
    const uptimeSeconds = uptimeElement.dataset.uptimeSeconds;
    if (uptimeSeconds && !isNaN(parseFloat(uptimeSeconds))) {
        // Convert seconds to days, hours, minutes, seconds
        const uptime = Math.floor(parseFloat(uptimeSeconds));
        const days = Math.floor(uptime / (60 * 60 * 24));
        const hours = Math.floor((uptime % (60 * 60 * 24)) / (60 * 60));
        const minutes = Math.floor((uptime % (60 * 60)) / 60);
        const seconds = Math.floor(uptime % 60);
        
        uptimeElement.textContent = `${days}d ${hours}h ${minutes}m ${seconds}s`;
        return; // Use the uptime seconds value and don't set up the interval
    }

    // Fall back to using start time if available
    const startTime = new Date(uptimeElement.dataset.startTime);
    if (isNaN(startTime.getTime())) {
        // Don't set error message here, as we'll get updates from WebSocket
        console.log('Invalid start time, waiting for WebSocket updates');
        return;
    }

    function updateUptimeDisplay() {
        const now = new Date();
        const diff = now - startTime;
        
        const days = Math.floor(diff / (1000 * 60 * 60 * 24));
        const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((diff % (1000 * 60)) / 1000);
        
        uptimeElement.textContent = `${days}d ${hours}h ${minutes}m ${seconds}s`;
    }

    updateUptimeDisplay();
    addInterval(updateUptimeDisplay, 1000);
}

// Track records by track_id for efficient updates
const recordsMap = new Map();

function initializeWebSocket() {
    try {
        // Remove any existing socket connection
        if (state.socket) {
            state.socket.disconnect();
        }

        // Check if io is defined
        if (typeof io === 'undefined') {
            console.error('Socket.IO not loaded');
            return;
        }

        // Initialize socket connection with better error handling
        state.socket = io({
            path: '/socket.io/',
            transports: ['websocket', 'polling'],
            upgrade: true,
            reconnection: true,
            reconnectionAttempts: CONFIG.maxReconnectAttempts,
            reconnectionDelay: CONFIG.reconnectDelay,
            timeout: 20000,
            forceNew: true
        });

        // Connection event handlers
        state.socket.on('connect', () => {
            console.log('Connected to server');
            state.reconnectAttempts = 0;
            state.connectionState = 'connected';
            updateConnectionStatus('Connected');
        });

        state.socket.on('disconnect', (reason) => {
            console.log('Disconnected from server:', reason);
            state.connectionState = 'disconnected';
            updateConnectionStatus('Disconnected');
            attemptReconnect();
        });

        state.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            state.reconnectAttempts++;
            state.connectionState = 'error';
            updateConnectionStatus('Connection Error');
            
            if (state.reconnectAttempts >= CONFIG.maxReconnectAttempts) {
                console.error('Max reconnection attempts reached');
                showError('Connection lost. Please refresh the page.');
            }
        });

        // Handle new records
        state.socket.on('new_records', (data) => {
            console.log('Received new records:', data);
            if (data && data.records) {
                handleNewRecords(data.records);
                if (data.pagination) {
                    try {
                        updatePagination(data.pagination);
                    } catch (error) {
                        console.error('Error updating pagination:', error);
                    }
                }
            }
        });

        // Handle updated records
        state.socket.on('updated_records', (data) => {
            console.log('Received updated records:', data);
            if (data && data.records) {
                handleUpdatedRecords(data.records);
                if (data.pagination) {
                    try {
                        updatePagination(data.pagination);
                    } catch (error) {
                        console.error('Error updating pagination:', error);
                    }
                }
            }
        });

        // Add error event handler
        state.socket.on('error', (error) => {
            console.error('Socket error:', error);
            showError('Connection error occurred. Attempting to reconnect...');
        });

        // Handle system metrics
        state.socket.on('system_metrics', (data) => {
            updateSystemMetrics(data);
        });

    } catch (error) {
        console.error('Error initializing WebSocket:', error);
        showError('Failed to initialize real-time updates. Please refresh the page.');
    }
}

// Record management
let records = new Map();
let currentPage = 1;
const recordsPerPage = 20;

// Add this helper function at the top level
function formatDateTime(datetimeStr) {
    try {
        const datetime = new Date(datetimeStr);
        if (isNaN(datetime.getTime())) {
            return 'N/A';
        }
        
        // Format: YYYY-MM-DD HH:mm:ss
        const year = datetime.getFullYear();
        const month = String(datetime.getMonth() + 1).padStart(2, '0');
        const day = String(datetime.getDate()).padStart(2, '0');
        const hours = String(datetime.getHours()).padStart(2, '0');
        const minutes = String(datetime.getMinutes()).padStart(2, '0');
        const seconds = String(datetime.getSeconds()).padStart(2, '0');
        
        return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
    } catch (error) {
        console.error('Error formatting date:', error);
        return 'N/A';
    }
}

// Record handling functions
function handleNewRecords(newRecords) {
    const tbody = document.querySelector('table tbody');
    if (!tbody) return;

    for (const record of newRecords) {
        // Create new row
        const row = createRecordRow(record);
        row.setAttribute('data-track-id', record.track_id);
        row.setAttribute('data-confidence', record.confidence || '0');
        
        // Insert at beginning of table
        tbody.insertBefore(row, tbody.firstChild);
        row.classList.add('highlight-new');

        // Remove last row if we're at the page limit
        if (tbody.children.length > CONFIG.maxRows) {
            tbody.removeChild(tbody.lastChild);
        }
    }

    // Check for duplicates after adding new records
    checkAndRemoveDuplicates();
}

function handleUpdatedRecords(updatedRecords) {
    const tbody = document.querySelector('table tbody');
    if (!tbody) return;

    for (const record of updatedRecords) {
        // Find existing row
        const existingRow = tbody.querySelector(`tr[data-track-id="${record.track_id}"]`);
        if (existingRow) {
            const existingConfidence = parseFloat(existingRow.getAttribute('data-confidence') || '0');
            const newConfidence = record.confidence || 0;

            // Only update if new confidence is higher
            if (newConfidence >= existingConfidence) {
                const newRow = createRecordRow(record);
                existingRow.innerHTML = newRow.innerHTML;
                existingRow.setAttribute('data-confidence', newConfidence);
            }
        }
    }

    // Check for duplicates after updating records
    checkAndRemoveDuplicates();
}

function createRecordRow(record) {
    const row = document.createElement('tr');
    const datetime = record.datetime ? formatDateTime(record.datetime) : 'N/A';
    
    row.innerHTML = `
        <td>${datetime}</td>
        <td>${record.ocr_output || 'N/A'}</td>
        <td>${record.camera_id}</td>
        <td>
            ${record.image_path ? `
                <a href="/images/${record.image_path.split('/').pop()}" target="_blank" class="btn btn-sm btn-outline-dark">
                    <i class="bi bi-image"></i> View
                </a>
            ` : `
                <span class="text-muted" title="Image not available">
                    <i class="bi bi-image-fill text-muted"></i> Unavailable
                </span>
            `}
        </td>
    `;
    
    return row;
}

// Create connection status element
function createConnectionStatus() {
    let statusElement = document.querySelector('.connection-status');
    if (!statusElement) {
        statusElement = document.createElement('div');
        statusElement.className = 'connection-status';
        document.body.appendChild(statusElement);
    }
    return statusElement;
}

function updateConnectionStatus(status) {
    const statusElement = createConnectionStatus();
    let statusText = '';
    
    switch (status) {
        case 'Connected':
            statusText = 'Connected';
            statusElement.style.backgroundColor = 'var(--success-color)';
            break;
        case 'Disconnected':
            statusText = 'Disconnected - Reconnecting...';
            statusElement.style.backgroundColor = 'var(--warning-color)';
            break;
        case 'Connection Error':
            statusText = 'Connection Error';
            statusElement.style.backgroundColor = 'var(--danger-color)';
            break;
    }
    statusElement.textContent = statusText;
    statusElement.className = `connection-status ${status.toLowerCase()}`;

    if (status === 'Connected') {
        setTimeout(() => {
            statusElement.style.opacity = '0';
        }, 5000);
    } else {
        statusElement.style.opacity = '1';
    }
}

// Update pagination function
function updatePagination(pagination) {
    const paginationElement = document.querySelector('#pagination');
    if (!paginationElement) {
        console.warn('Pagination element not found, skipping pagination update');
        return;
    }

    if (!pagination || !pagination.total_pages) {
        console.warn('Invalid pagination data received');
        paginationElement.innerHTML = '';
        return;
    }

    const totalPages = pagination.total_pages;
    const currentPage = pagination.page || 1;
    
    let html = '';
    if (totalPages > 1) {
        html = `<li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
            <a class="page-link" href="#" onclick="changePage(1); return false;" aria-label="First">
                <span aria-hidden="true">&laquo;</span>
            </a>
        </li>`;
        
        html += `<li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
            <a class="page-link" href="#" onclick="changePage(${currentPage - 1}); return false;" aria-label="Previous">
                <span aria-hidden="true">&lsaquo;</span>
            </a>
        </li>`;
        
        for (let i = Math.max(1, currentPage - 2); i <= Math.min(totalPages, currentPage + 2); i++) {
            html += `<li class="page-item ${i === currentPage ? 'active' : ''}">
                <a class="page-link" href="#" onclick="changePage(${i}); return false;">${i}</a>
            </li>`;
        }
        
        html += `<li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
            <a class="page-link" href="#" onclick="changePage(${currentPage + 1}); return false;" aria-label="Next">
                <span aria-hidden="true">&rsaquo;</span>
            </a>
        </li>`;
        
        html += `<li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
            <a class="page-link" href="#" onclick="changePage(${totalPages}); return false;" aria-label="Last">
                <span aria-hidden="true">&raquo;</span>
            </a>
        </li>`;
    }
    
    paginationElement.innerHTML = html;

    // Update record count display if it exists
    const recordCountElement = document.querySelector('.text-muted');
    if (recordCountElement && pagination.total !== undefined) {
        const start = ((currentPage - 1) * pagination.per_page) + 1;
        const end = Math.min(currentPage * pagination.per_page, pagination.total);
        recordCountElement.textContent = `Showing ${start} to ${end} of ${pagination.total} entries`;
    }
}

async function changePage(page) {
    currentPage = page;
    await loadRecords(page);
}

async function loadInitialRecords() {
    await loadRecords(1);
}

// Add the missing insertRecordIntoTable function
function insertRecordIntoTable(record, animate = true) {
    const tbody = document.querySelector('#records-table tbody');
    if (!tbody) return;

    const row = createRecordRow(record);
    row.setAttribute('data-track-id', record.track_id);
    row.setAttribute('data-confidence', record.confidence || '0');
    
    tbody.appendChild(row);
    if (animate) {
        row.classList.add('highlight-new');
    }
}

// Update loadRecords function to properly handle records
async function loadRecords(page) {
    try {
        showLoading();
        const response = await fetch(`/api/records?page=${page}&per_page=${recordsPerPage}`);
        if (!response.ok) throw new Error('Failed to fetch records');
        
        const data = await response.json();
        const tbody = document.querySelector('#records-table tbody');
        tbody.innerHTML = '';
        
        records.clear();
        if (data.records && Array.isArray(data.records)) {
            data.records.forEach(record => {
                records.set(record.track_id, record);
                insertRecordIntoTable(record, false);
            });
        }
        
        updatePagination(data.pagination);
        hideLoading();
        
    } catch (error) {
        console.error('Error loading records:', error);
        hideLoading();
        const errorDiv = document.getElementById('error-message');
        if (errorDiv) {
            errorDiv.textContent = 'Error loading records. Please try again later.';
            errorDiv.style.display = 'block';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }
    }
}

// Update the CSS styles
const style = document.createElement('style');
style.textContent = `
    /* Animation for new records only */
    @keyframes highlightNew {
        0% { background-color: rgba(255, 215, 0, 0.3); }  /* Golden color with 70% transparency */
        100% { background-color: transparent; }
    }

    .highlight-new {
        animation: highlightNew 2s ease-out;
    }

    /* Center align all table content */
    #records-table th,
    #records-table td {
        text-align: center !important;
        vertical-align: middle !important;
    }

    /* Pagination styles */
    .pagination {
        margin-bottom: 0;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .pagination .page-item.active .page-link {
        background-color: #007bff;
        border-color: #007bff;
    }

    .pagination .page-link {
        padding: 0.5rem 0.75rem;
        margin: 0 2px;
        color: #007bff;
        background-color: #fff;
        border: 1px solid #dee2e6;
    }

    .pagination .page-item.disabled .page-link {
        color: #6c757d;
        pointer-events: none;
        background-color: #fff;
        border-color: #dee2e6;
    }
`;
document.head.appendChild(style);

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('Document loaded, initializing application...');
    initializeUI();
    initializeUptimeCounter();
    initializeWebSocket();
    startDuplicateCheck();
    
    // Create error message div if it doesn't exist
    if (!document.getElementById('error-message')) {
        const errorDiv = document.createElement('div');
        errorDiv.id = 'error-message';
        errorDiv.className = 'error-message';
        document.body.appendChild(errorDiv);
    }
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (state.socket) {
        state.socket.disconnect();
    }
});

// Date filtering setup
function setupDateFiltering() {
    const dateFilteringCheckbox = document.getElementById('dateFiltering');
    const dateRangeContainer = document.getElementById('dateRangeContainer');
    
    if (dateFilteringCheckbox && dateRangeContainer) {
        dateFilteringCheckbox.addEventListener('change', function() {
            if (this.checked) {
                dateRangeContainer.style.display = 'block';
            } else {
                dateRangeContainer.style.display = 'none';
                document.getElementById('start_date').value = '';
                document.getElementById('end_date').value = '';
                this.form.submit();
            }
        });

        // Initialize daterangepicker if enabled
        if (dateFilteringCheckbox.checked) {
            dateRangeContainer.style.display = 'block';
            initializeDateRangePicker();
        }
    }
}

// Initialize date range picker
function initializeDateRangePicker() {
    $('#daterange').daterangepicker({
        startDate: moment().subtract(29, 'days'),
        endDate: moment(),
        ranges: {
           'Today': [moment(), moment()],
           'Yesterday': [moment().subtract(1, 'days'), moment().subtract(1, 'days')],
           'Last 7 Days': [moment().subtract(6, 'days'), moment()],
           'Last 30 Days': [moment().subtract(29, 'days'), moment()],
           'This Month': [moment().startOf('month'), moment().endOf('month')],
           'Last Month': [moment().subtract(1, 'month').startOf('month'), moment().subtract(1, 'month').endOf('month')]
        }
    }, function(start, end) {
        $('#start_date').val(start.format('YYYY-MM-DD'));
        $('#end_date').val(end.format('YYYY-MM-DD'));
        $('#filterForm').submit();
    });
}

// Image error handling
function handleImageError(img) {
    img.style.display = 'none';
    document.getElementById('imageError').classList.remove('d-none');
}

// Modal cleanup
$('#imageModal').on('hidden.bs.modal', function () {
    const img = document.getElementById('previewImage');
    img.src = '';
    img.style.display = 'block';
    document.getElementById('imageError').classList.add('d-none');
});

// Add a function to check for duplicate track IDs
function checkAndRemoveDuplicates() {
    const tbody = document.querySelector('#records-table tbody');
    if (!tbody) return;

    const trackIdMap = new Map(); // Map to store track_id -> {row, confidence}

    // First pass: collect all rows by track_id
    Array.from(tbody.querySelectorAll('tr[data-track-id]')).forEach(row => {
        const trackId = row.getAttribute('data-track-id');
        const confidence = parseFloat(row.getAttribute('data-confidence') || '0');
        
        if (trackIdMap.has(trackId)) {
            const existing = trackIdMap.get(trackId);
            if (confidence > existing.confidence) {
                // Remove the existing row with lower confidence
                existing.row.remove();
                trackIdMap.set(trackId, { row, confidence });
            } else {
                // Remove the current row with lower confidence
                row.remove();
            }
        } else {
            trackIdMap.set(trackId, { row, confidence });
        }
    });
}

// Add interval to check for duplicates
function startDuplicateCheck() {
    return setInterval(checkAndRemoveDuplicates, 1000);
}

// System Status WebSocket
let socket;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

function updateSystemMetrics(data) {
    // Update CPU Usage
    const cpuElement = document.querySelector('#metricsCollapse .metric-item:nth-child(2) p');
    if (cpuElement) cpuElement.textContent = `${data.cpu_usage}%`;

    // Update Memory Usage
    const memoryElement = document.querySelector('#metricsCollapse .metric-item:nth-child(3) p');
    if (memoryElement) memoryElement.textContent = `${data.memory_usage}%`;

    // Update Uptime if available
    if (data.uptime !== undefined) {
        const uptimeElement = document.getElementById('system-uptime');
        if (uptimeElement) {
            // Convert seconds to days, hours, minutes, seconds
            const uptime = Math.floor(data.uptime);
            const days = Math.floor(uptime / (60 * 60 * 24));
            const hours = Math.floor((uptime % (60 * 60 * 24)) / (60 * 60));
            const minutes = Math.floor((uptime % (60 * 60)) / 60);
            const seconds = Math.floor(uptime % 60);
            
            uptimeElement.textContent = `${days}d ${hours}h ${minutes}m ${seconds}s`;
        }
    }

    // Update Camera Status
    const cameraStatusContainer = document.querySelector('.camera-status-grid');
    if (cameraStatusContainer && data.camera_status) {
        cameraStatusContainer.innerHTML = '';
        data.camera_status.forEach(camera => {
            const div = document.createElement('div');
            div.className = `camera-status-item ${camera.status ? 'active' : 'inactive'}`;
            div.textContent = `${camera.id}: ${camera.status ? 'Online' : 'Offline'}`;
            cameraStatusContainer.appendChild(div);
        });
    }
}

function attemptReconnect() {
    if (reconnectAttempts < maxReconnectAttempts) {
        reconnectAttempts++;
        setTimeout(() => {
            console.log(`Attempting to reconnect (${reconnectAttempts}/${maxReconnectAttempts})`);
            socket.connect();
        }, 5000);
    }
} 
