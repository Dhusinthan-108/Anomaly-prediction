/**
 * 🚦 Traffic Anomaly Detection System - Frontend JavaScript
 * Handles video upload, processing, and result visualization
 */

// =========================================
// Configuration
// =========================================
const API_BASE_URL = 'http://localhost:5000/api';
let currentFileId = null;
let processingInterval = null;

// =========================================
// DOM Elements
// =========================================
const videoInput = document.getElementById('video-input');
const uploadArea = document.getElementById('upload-area');
const videoPreview = document.getElementById('video-preview');
const previewVideo = document.getElementById('preview-video');
const videoFilename = document.getElementById('video-filename');
const changeVideoBtn = document.getElementById('change-video-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const processingStatus = document.getElementById('processing-status');
const processingMessage = document.getElementById('processing-message');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const resultsSection = document.querySelector('.results-section');
const outputVideo = document.getElementById('output-video');
const downloadBtn = document.getElementById('download-btn');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toast-message');

// =========================================
// Event Listeners
// =========================================
videoInput.addEventListener('change', handleVideoSelect);
changeVideoBtn.addEventListener('click', resetUpload);
analyzeBtn.addEventListener('click', startAnalysis);
downloadBtn.addEventListener('click', downloadResult);

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleVideoFile(files[0]);
    }
});

// =========================================
// Video Selection & Preview
// =========================================
function handleVideoSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleVideoFile(file);
    }
}

function handleVideoFile(file) {
    // Validate file type
    if (!file.type.startsWith('video/')) {
        showToast('❌ Please select a valid video file', 'error');
        return;
    }

    // Validate file size (max 500MB)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast('❌ File size exceeds 500MB limit', 'error');
        return;
    }

    // Create preview URL
    const videoURL = URL.createObjectURL(file);
    previewVideo.src = videoURL;
    videoFilename.textContent = file.name;

    // Show preview, hide upload area
    uploadArea.classList.add('hidden');
    videoPreview.classList.remove('hidden');

    showToast('✅ Video loaded successfully', 'success');
}

function resetUpload() {
    videoInput.value = '';
    previewVideo.src = '';
    uploadArea.classList.remove('hidden');
    videoPreview.classList.add('hidden');
    processingStatus.classList.add('hidden');
    currentFileId = null;
}

// =========================================
// Video Upload & Analysis
// =========================================
async function uploadVideo(file) {
    const formData = new FormData();
    formData.append('video', file);

    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        const data = await response.json();
        return data.file_id;
    } catch (error) {
        console.error('Upload error:', error);
        throw error;
    }
}

async function processVideo(fileId) {
    try {
        const response = await fetch(`${API_BASE_URL}/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ file_id: fileId })
        });

        if (!response.ok) {
            throw new Error('Processing failed');
        }

        const data = await response.json();
        return data.results;
    } catch (error) {
        console.error('Processing error:', error);
        throw error;
    }
}

async function checkStatus(fileId) {
    try {
        const response = await fetch(`${API_BASE_URL}/status/${fileId}`);

        if (!response.ok) {
            // If 404, the status might not be initialized yet - return a default status
            if (response.status === 404) {
                console.warn('Status not found yet, waiting for initialization...');
                return {
                    status: 'processing',
                    progress: 0,
                    message: 'Initializing...'
                };
            }
            throw new Error(`Status check failed: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Status check error:', error);
        // Return a default status instead of throwing to keep polling alive
        return {
            status: 'processing',
            progress: 0,
            message: 'Connecting to server...'
        };
    }
}

async function startAnalysis() {
    try {
        // Get the video file
        const file = videoInput.files[0];
        if (!file) {
            showToast('❌ No video selected', 'error');
            return;
        }

        // Show processing status
        videoPreview.classList.add('hidden');
        processingStatus.classList.remove('hidden');
        processingMessage.textContent = 'Uploading video...';
        updateProgress(0);

        // Upload video
        currentFileId = await uploadVideo(file);
        showToast('✅ Video uploaded', 'success');

        // Start processing
        processingMessage.textContent = 'Processing video...';
        updateProgress(10);

        // Start processing in background
        processVideo(currentFileId).catch(error => {
            console.error('Background processing error:', error);
        });

        // Poll for status updates with retry limit
        let retryCount = 0;
        const maxRetries = 300; // 5 minutes at 1 second intervals

        processingInterval = setInterval(async () => {
            const status = await checkStatus(currentFileId);

            retryCount++;

            // Check if we've exceeded max retries
            if (retryCount > maxRetries) {
                clearInterval(processingInterval);
                showToast('❌ Processing timeout. Please try again.', 'error');
                resetUpload();
                return;
            }

            if (status.status === 'processing' || status.status === 'queued' || status.status === 'uploaded' || status.status === 'waiting') {
                processingMessage.textContent = status.message;
                updateProgress(status.progress * 100);
            } else if (status.status === 'completed') {
                clearInterval(processingInterval);
                updateProgress(100);
                showResults(status.results);
            } else if (status.status === 'error') {
                clearInterval(processingInterval);
                showToast(`❌ Error: ${status.message}`, 'error');
                resetUpload();
            }
        }, 1000);

    } catch (error) {
        console.error('Analysis error:', error);
        showToast('❌ Analysis failed. Please try again.', 'error');
        resetUpload();
    }
}

function updateProgress(percentage) {
    progressFill.style.width = `${percentage}%`;
    progressText.textContent = `${Math.round(percentage)}%`;
}

// =========================================
// Results Display
// =========================================
function showResults(results) {
    // Hide processing, show results
    processingStatus.classList.add('hidden');
    resultsSection.classList.remove('hidden');

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    // Update stats with null checks
    const statAnomalies = document.getElementById('stat-anomalies');
    const statRatio = document.getElementById('stat-ratio');
    const statDuration = document.getElementById('stat-duration');

    if (statAnomalies) {
        statAnomalies.textContent = results.anomaly_count || 0;
    }

    if (statRatio) {
        const ratio = results.anomaly_ratio || 0;
        statRatio.textContent = `${(ratio * 100).toFixed(1)}%`;
    }

    if (statDuration && results.video_info) {
        statDuration.textContent = `${(results.video_info.duration || 0).toFixed(1)}s`;
    }

    // Set output video
    outputVideo.src = `${API_BASE_URL}/video/${currentFileId}`;

    // Display event types
    displayEventTypes(results.event_types || {});

    // Display timeline
    displayTimeline(results.anomaly_frames || [], results.total_frames || 0);

    showToast('✅ Analysis complete!', 'success');
}

function displayEventTypes(eventTypes) {
    const eventsList = document.getElementById('events-list');
    eventsList.innerHTML = '';

    if (Object.keys(eventTypes).length === 0) {
        eventsList.innerHTML = '<p style="color: var(--text-secondary); text-align: center;">No anomalies detected</p>';
        return;
    }

    // Sort by count
    const sortedEvents = Object.entries(eventTypes).sort((a, b) => b[1] - a[1]);

    sortedEvents.forEach(([eventName, count]) => {
        const eventItem = document.createElement('div');
        eventItem.className = 'event-item';
        eventItem.innerHTML = `
            <span class="event-name">${eventName}</span>
            <span class="event-count">${count}</span>
        `;
        eventsList.appendChild(eventItem);
    });
}

function displayTimeline(anomalyFrames, totalFrames) {
    const timelineChart = document.getElementById('timeline-chart');
    timelineChart.innerHTML = '';

    // Create simple timeline visualization
    const timeline = document.createElement('div');
    timeline.style.cssText = `
        display: flex;
        height: 100%;
        width: 100%;
        background: #f3f4f6; /* Light gray */
        position: relative;
        overflow: hidden;
    `;

    // Create markers for anomalies
    anomalyFrames.forEach(frameNum => {
        const position = (frameNum / totalFrames) * 100;
        const marker = document.createElement('div');
        marker.style.cssText = `
            position: absolute;
            left: ${position}%;
            width: 3px;
            height: 100%;
            background: #ef4444; /* Red */
            opacity: 0.8;
        `;
        marker.title = `Anomaly at frame ${frameNum}`;
        timeline.appendChild(marker);
    });

    timelineChart.appendChild(timeline);

    // Add labels outside the chart or rely on the container
    // The previous implementation added labels inside the chart container which might break layout
    // We will leave labels out for simplicity as per "minimal" request, or add them below if needed.
    // Given the new design has a dedicated card for it, we can keep it simple.
}

// =========================================
// Download Result
// =========================================
function downloadResult() {
    if (currentFileId) {
        window.open(`${API_BASE_URL}/download/${currentFileId}`, '_blank');
        showToast('⬇️ Download started', 'success');
    }
}

// =========================================
// Toast Notifications
// =========================================
function showToast(message, type = 'info') {
    toastMessage.textContent = message;
    toast.classList.remove('hidden');

    // Auto-hide after 3 seconds
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}

// =========================================
// Smooth Scrolling for Navigation
// =========================================
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href');
        const targetSection = document.querySelector(targetId);

        if (targetSection) {
            targetSection.scrollIntoView({ behavior: 'smooth' });

            // Update active link
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        }
    });
});

// =========================================
// Initialize
// =========================================
console.log('🚦 Traffic Anomaly Detection System initialized');
console.log('API Base URL:', API_BASE_URL);
