import { loadCameras, registerCamera, deleteCamera, startCamera, stopCamera } from './cameras.js';
import { connectStream, disconnectStream } from './stream.js';

const tabs = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

let currentStreamCameraId = null;
let refreshInterval = null;

function switchTab(tabName) {
    tabs.forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tabName));
    tabContents.forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}-tab`);
    });

    if (tabName !== 'stream-viewer' && currentStreamCameraId) {
        disconnectStream();
        currentStreamCameraId = null;
    }
}

tabs.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

document.getElementById('add-camera-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = document.getElementById('cam-name').value.trim();
    const url = document.getElementById('cam-url').value.trim();
    const msgEl = document.getElementById('form-msg');

    if (!name || !url) return;

    try {
        await registerCamera(name, url);
        msgEl.textContent = 'Camera registered successfully!';
        msgEl.className = 'form-msg success';
        document.getElementById('add-camera-form').reset();
        setTimeout(() => { msgEl.textContent = ''; }, 3000);
        switchTab('cameras');
    } catch (err) {
        msgEl.textContent = err.message || 'Failed to register camera';
        msgEl.className = 'form-msg error';
    }
});

document.getElementById('btn-test-camera').addEventListener('click', async () => {
    const url = document.getElementById('cam-url').value.trim();
    const msgEl = document.getElementById('form-msg');
    const previewArea = document.getElementById('preview-area');
    const previewImg = document.getElementById('preview-img');
    const previewStats = document.getElementById('preview-stats');

    if (!url) {
        msgEl.textContent = 'Please enter a stream URL first';
        msgEl.className = 'form-msg error';
        return;
    }

    msgEl.textContent = 'Testing camera...';
    msgEl.className = 'form-msg';

    try {
        const res = await fetch('/api/cameras/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: 'test', stream_url: url }),
        });
        const data = await res.json();

        if (data.success) {
            previewImg.src = data.frame;
            previewStats.innerHTML = `
                Resolution: ${data.resolution.width}x${data.resolution.height} |
                Faces: ${data.faces_detected} |
                Objects: ${data.objects_detected}
            `;
            previewArea.classList.remove('hidden');
            msgEl.textContent = 'Camera test successful!';
            msgEl.className = 'form-msg success';
        } else {
            previewArea.classList.add('hidden');
            msgEl.textContent = data.error || 'Camera test failed';
            msgEl.className = 'form-msg error';
        }
    } catch (err) {
        previewArea.classList.add('hidden');
        msgEl.textContent = 'Failed to test camera: ' + err.message;
        msgEl.className = 'form-msg error';
    }
});

document.getElementById('btn-close-viewer').addEventListener('click', () => {
    switchTab('cameras');
});

document.getElementById('btn-reconnect').addEventListener('click', () => {
    if (currentStreamCameraId) {
        connectStream(currentStreamCameraId);
    }
});

document.getElementById('btn-copy-url').addEventListener('click', () => {
    const urlInput = document.getElementById('stream-ws-url');
    urlInput.select();
    navigator.clipboard.writeText(urlInput.value);
});

document.getElementById('btn-copy-mjpeg-url').addEventListener('click', () => {
    const urlInput = document.getElementById('stream-mjpeg-url');
    urlInput.select();
    navigator.clipboard.writeText(urlInput.value);
});

window.viewCamera = (cameraId, cameraName) => {
    currentStreamCameraId = cameraId;
    document.getElementById('viewer-title').textContent = cameraName;
    switchTab('stream-viewer');
    connectStream(cameraId);
};

window.deleteCameraAction = async (cameraId) => {
    if (!confirm('Delete this camera?')) return;
    await deleteCamera(cameraId);
    loadCameras();
};

window.toggleCamera = async (cameraId, isActive) => {
    if (isActive) {
        await stopCamera(cameraId);
    } else {
        await startCamera(cameraId);
    }
    loadCameras();
};

async function checkHealth() {
    const badge = document.getElementById('health-indicator');
    try {
        const res = await fetch('/api/health');
        if (res.ok) {
            badge.textContent = 'Online';
            badge.className = 'health-badge ok';
        } else {
            badge.textContent = 'Error';
            badge.className = 'health-badge error';
        }
    } catch {
        badge.textContent = 'Offline';
        badge.className = 'health-badge error';
    }
}

async function init() {
    await checkHealth();
    await loadCameras();
    refreshInterval = setInterval(async () => {
        await checkHealth();
        await loadCameras();
    }, 5000);
}

init();
