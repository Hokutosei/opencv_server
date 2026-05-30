const canvas = document.getElementById('stream-canvas');
const ctx = canvas.getContext('2d');
const overlay = document.getElementById('stream-overlay');
const streamImg = document.getElementById('stream-img');

let ws = null;
let useMjpeg = true;  // prefer MJPEG for video, WS for metadata

export function connectStream(cameraId) {
    disconnectStream();
    overlay.classList.add('hidden');

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/stream/${cameraId}`;
    const mjpegUrl = `${location.protocol}//${location.host}/stream/mjpeg/${cameraId}`;

    document.getElementById('stream-ws-url').value = wsUrl;
    document.getElementById('stream-mjpeg-url').value = mjpegUrl;

    if (useMjpeg) {
        // Use MJPEG <img> for video - most reliable browser method
        streamImg.src = mjpegUrl;
        streamImg.style.display = 'block';
        canvas.style.display = 'none';

        streamImg.onerror = () => {
            overlay.classList.remove('hidden');
            overlay.querySelector('span').textContent = 'Stream error - try reconnect';
        };
    } else {
        // Fallback: WebSocket + canvas
        streamImg.style.display = 'none';
        canvas.style.display = 'block';
    }

    // WebSocket for metadata (detection stats)
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
            if (!useMjpeg) {
                // Only use WS frames when not using MJPEG
                const blob = new Blob([event.data], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                const img = new Image();
                img.onload = () => {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                    URL.revokeObjectURL(url);
                };
                img.src = url;
            }
        } else {
            try {
                const meta = JSON.parse(event.data);
                if (meta.heartbeat) return;
                if (meta.status === 'offline') {
                    overlay.classList.remove('hidden');
                    return;
                }
                if (meta.error) {
                    overlay.classList.remove('hidden');
                    overlay.querySelector('span').textContent = meta.error === 'camera_not_found' ? 'Camera not found' : 'Error';
                    return;
                }
                updateStats(meta);
            } catch {
                // ignore parse errors
            }
        }
    };

    ws.onclose = () => {
        // If WS closes but MJPEG is working, don't show overlay
        if (!useMjpeg || !streamImg.src) {
            overlay.classList.remove('hidden');
        }
    };

    ws.onerror = () => {
        if (!useMjpeg) {
            overlay.classList.remove('hidden');
        }
    };
}

export function disconnectStream() {
    if (ws) {
        ws.close();
        ws = null;
    }
    if (streamImg) {
        streamImg.src = '';
        streamImg.style.display = 'none';
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.style.display = 'block';
    document.getElementById('stat-fps').textContent = '--';
    document.getElementById('stat-faces').textContent = '--';
    document.getElementById('stat-objects').textContent = '--';
}

function updateStats(meta) {
    document.getElementById('stat-fps').textContent = meta.fps !== undefined ? meta.fps.toFixed(1) : '--';
    document.getElementById('stat-faces').textContent = meta.face_count !== undefined ? meta.face_count : '--';
    document.getElementById('stat-objects').textContent = meta.object_count !== undefined ? meta.object_count : '--';
}
