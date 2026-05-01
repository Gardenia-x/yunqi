// MP3 → MIDI 转换器 - 前端脚本

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const convertBtn = document.getElementById('convertBtn');
const versionSelect = document.getElementById('versionSelect');
const progressSection = document.getElementById('progressSection');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');
const progressFill = document.getElementById('progressFill');

let selectedFile = null;

// ======== 文件选择 ========

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

function handleFile(file) {
    const validExts = ['.mp3', '.wav', '.m4a', '.flac', '.ogg'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!validExts.includes(ext)) {
        showError('不支持的文件格式，请上传 MP3 / WAV / M4A / FLAC / OGG');
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        showError('文件过大，最大支持 50MB');
        return;
    }

    selectedFile = file;
    fileInfo.style.display = 'block';
    fileInfo.textContent = `${file.name} (${formatSize(file.size)})`;
    convertBtn.disabled = false;
    hideError();
}

function formatSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// ======== 转换 ========

convertBtn.addEventListener('click', convert);

async function convert() {
    if (!selectedFile) return;

    // 显示进度
    progressSection.style.display = 'block';
    progressFill.classList.add('active');
    resultSection.style.display = 'none';
    errorSection.style.display = 'none';
    convertBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('version', versionSelect.value);

    try {
        const response = await fetch('/api/convert', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || `服务器错误 (${response.status})`);
        }

        // 显示结果
        document.getElementById('resultFilename').textContent = data.filename;
        document.getElementById('resultVersion').textContent = data.version;
        document.getElementById('resultNotes').textContent = data.note_count + ' 个';
        document.getElementById('resultTime').textContent = data.elapsed_seconds + ' 秒';
        document.getElementById('downloadLink').href = data.download_url;

        resultSection.style.display = 'block';
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (error) {
        showError(error.message);
    } finally {
        progressSection.style.display = 'none';
        progressFill.classList.remove('active');
        convertBtn.disabled = false;
    }
}

// ======== 重置 ========

function resetForm() {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.style.display = 'none';
    convertBtn.disabled = true;
    resultSection.style.display = 'none';
    errorSection.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ======== 错误处理 ========

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideError() {
    errorSection.style.display = 'none';
}

// ======== 键盘快捷键 ========

document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') convert();
    if (e.key === 'Escape') resetForm();
});
