const cameraList = document.getElementById('camera-list');

export async function loadCameras() {
    try {
        const res = await fetch('/api/cameras');
        const cameras = await res.json();
        renderCameras(cameras);
    } catch (err) {
        cameraList.innerHTML = '<p class="empty-msg">Failed to load cameras.</p>';
    }
}

export async function registerCamera(name, streamUrl) {
    const res = await fetch('/api/cameras', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, stream_url: streamUrl }),
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to register');
    }
    return res.json();
}

export async function deleteCamera(cameraId) {
    const res = await fetch(`/api/cameras/${cameraId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete');
    return res.json();
}

export async function updateCamera(cameraId, name, streamUrl) {
    const body = {};
    if (name !== undefined) body.name = name;
    if (streamUrl !== undefined) body.stream_url = streamUrl;
    const res = await fetch(`/api/cameras/${cameraId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to update');
    }
    return res.json();
}

export async function startCamera(cameraId) {
    const res = await fetch(`/api/cameras/${cameraId}/start`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to start');
    return res.json();
}

export async function stopCamera(cameraId) {
    const res = await fetch(`/api/cameras/${cameraId}/stop`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to stop');
    return res.json();
}

function renderCameras(cameras) {
    if (!cameras || cameras.length === 0) {
        cameraList.innerHTML = '<p class="empty-msg">No cameras registered. Add one to get started.</p>';
        return;
    }

    cameraList.innerHTML = cameras.map(cam => {
        const statusClass = cam.is_online ? 'online' : 'offline';
        const statusText = cam.is_online ? 'Online' : 'Offline';
        const fps = cam.stats ? cam.stats.current_fps.toFixed(1) : '0.0';
        const faces = cam.stats ? cam.stats.total_faces : 0;
        const objects = cam.stats ? cam.stats.total_objects : 0;
        const wsUrl = cam.stream_ws_url || '';
        const mjpegUrl = `${location.protocol}//${location.host}/stream/mjpeg/${cam.id}`;
        const toggleLabel = cam.is_active ? 'Stop' : 'Start';

        return `
            <div class="camera-card" data-camera-id="${cam.id}">
                <div class="camera-card-header">
                    <h3>
                        <span class="status-dot ${statusClass}"></span>
                        ${escHtml(cam.name)}
                    </h3>
                    <span style="font-size:0.75rem;color:var(--text-dim)">${statusText}</span>
                </div>
                <div class="camera-card-url">${escHtml(cam.stream_url)}</div>
                <div class="camera-card-stats">
                    <span>FPS: <strong>${fps}</strong></span>
                    <span>Faces: <strong>${faces}</strong></span>
                    <span>Objects: <strong>${objects}</strong></span>
                </div>
                <div class="camera-card-urls">
                    <div class="url-label">WS: <code>${escHtml(wsUrl)}</code></div>
                    <div class="url-label">MJPEG: <code>${escHtml(mjpegUrl)}</code></div>
                </div>
                <div class="camera-card-actions">
                    <button class="btn-primary" onclick="viewCamera('${cam.id}', '${escAttr(cam.name)}')">View</button>
                    <a class="btn-secondary" href="${escAttr(mjpegUrl)}" target="_blank" style="text-decoration:none;display:inline-flex;align-items:center;padding:6px 14px;">Open MJPEG</a>
                    <button class="btn-secondary" onclick="editCamera('${cam.id}', '${escAttr(cam.name)}', '${escAttr(cam.stream_url)}')">Edit</button>
                    <button class="btn-secondary" onclick="toggleCamera('${cam.id}', ${cam.is_active})">${toggleLabel}</button>
                    <button class="btn-danger" onclick="deleteCameraAction('${cam.id}')">Delete</button>
                </div>
            </div>
        `;
    }).join('');
}

function escHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escAttr(str) {
    return str.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}
