const canvas = document.getElementById('stream-canvas');
const ctx = canvas.getContext('2d');
const overlay = document.getElementById('stream-overlay');

let ws = null;

export function connectStream(cameraId) {
    disconnectStream();
    overlay.classList.add('hidden');

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/stream/${cameraId}`;

    document.getElementById('stream-ws-url').value = wsUrl;

    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
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
        overlay.classList.remove('hidden');
    };

    ws.onerror = () => {
        overlay.classList.remove('hidden');
    };
}

export function disconnectStream() {
    if (ws) {
        ws.close();
        ws = null;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('stat-fps').textContent = '--';
    document.getElementById('stat-faces').textContent = '--';
    document.getElementById('stat-objects').textContent = '--';
}

function updateStats(meta) {
    document.getElementById('stat-fps').textContent = meta.fps !== undefined ? meta.fps.toFixed(1) : '--';
    document.getElementById('stat-faces').textContent = meta.face_count !== undefined ? meta.face_count : '--';
    document.getElementById('stat-objects').textContent = meta.object_count !== undefined ? meta.object_count : '--';
}
