// MIDI评分系统 - 前端交互脚本

// 全局变量
let currentSessionId = null;
let analysisResults = null;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // 文件选择事件监听
    document.getElementById('reference_file').addEventListener('change', function(e) {
        updateFileInfo('reference_info', this.files[0]);
    });

    document.getElementById('test_file').addEventListener('change', function(e) {
        updateFileInfo('test_info', this.files[0]);
    });

    // 初始化工具提示
    initTooltips();
});

// 更新文件信息显示
function updateFileInfo(elementId, file) {
    const element = document.getElementById(elementId);
    if (file) {
        element.textContent = `${file.name} (${formatFileSize(file.size)})`;
        element.classList.add('selected');
    } else {
        element.textContent = '未选择文件';
        element.classList.remove('selected');
    }
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 分析文件
async function analyzeFiles() {
    const referenceFile = document.getElementById('reference_file').files[0];
    const testFile = document.getElementById('test_file').files[0];

    // 验证文件
    if (!referenceFile || !testFile) {
        showError('请选择参考文件和测试文件');
        return;
    }

    // 验证文件类型
    if (!isMidiFile(referenceFile) || !isMidiFile(testFile)) {
        showError('只支持MIDI文件 (.mid, .midi)');
        return;
    }

    // 验证文件大小（最大10MB）
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (referenceFile.size > maxSize || testFile.size > maxSize) {
        showError('文件大小不能超过10MB');
        return;
    }

    // 准备表单数据
    const formData = new FormData();
    formData.append('reference_file', referenceFile);
    formData.append('test_file', testFile);

    // 显示进度条
    showProgress();
    updateProgressText('正在上传文件...');

    try {
        // 发送分析请求
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `服务器错误: ${response.status}`);
        }

        updateProgressText('正在分析MIDI数据...');
        simulateProgress(30, 70);

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        updateProgressText('正在生成可视化图表...');
        simulateProgress(70, 95);

        // 保存结果
        analysisResults = data;
        currentSessionId = data.session_id;

        // 更新UI显示结果
        updateResultsUI(data);

        // 加载图像
        await loadAnalysisImage(data.image_url);

        updateProgressText('分析完成！');
        simulateProgress(95, 100);

        // 显示结果区域
        setTimeout(() => {
            hideProgress();
            showResults();
        }, 1000);

    } catch (error) {
        console.error('分析错误:', error);
        hideProgress();
        showError(error.message || '分析失败，请重试');
    }
}

// 模拟进度条动画
function simulateProgress(start, end) {
    const progressFill = document.getElementById('progressFill');
    let current = start;
    const interval = setInterval(() => {
        if (current >= end) {
            clearInterval(interval);
            return;
        }
        current += 1;
        progressFill.style.width = current + '%';
    }, 20);
}

// 更新进度文本
function updateProgressText(text) {
    document.getElementById('progressText').textContent = text;
}

// 显示进度条
function showProgress() {
    document.getElementById('progressContainer').style.display = 'block';
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('resetBtn').disabled = true;
}

// 隐藏进度条
function hideProgress() {
    document.getElementById('progressContainer').style.display = 'none';
    document.getElementById('analyzeBtn').disabled = false;
    document.getElementById('resetBtn').disabled = false;
}

// 更新结果UI
function updateResultsUI(data) {
    // 更新准确度指标
    if (data.accuracy) {
        document.getElementById('precision').textContent = data.accuracy.precision.toFixed(3);
        document.getElementById('recall').textContent = data.accuracy.recall.toFixed(3);
        document.getElementById('f1').textContent = data.accuracy.f1.toFixed(3);
        document.getElementById('onset_tol').textContent = data.accuracy.onset_tol;
        document.getElementById('pitch_tol').textContent = data.accuracy.pitch_tol;
    }

    // 更新节奏分析指标
    if (data.rhythm) {
        document.getElementById('dtw_distance').textContent = data.rhythm.dtw_distance.toFixed(1);
        document.getElementById('avg_deviation').textContent = data.rhythm.avg_time_deviation;
        document.getElementById('max_deviation').textContent = data.rhythm.max_deviation;
    }
}

// 加载分析图像
async function loadAnalysisImage(imageUrl) {
    const imageContainer = document.getElementById('imageContainer');

    try {
        // 创建图像元素
        const img = document.createElement('img');
        img.src = imageUrl;
        img.alt = 'MIDI分析可视化图表';
        img.onload = function() {
            // 清空容器并添加图像
            imageContainer.innerHTML = '';
            imageContainer.appendChild(img);
        };

        // 添加加载错误处理
        img.onerror = function() {
            imageContainer.innerHTML = `
                <div class="image-error">
                    <i class="fas fa-exclamation-circle"></i>
                    <p>无法加载可视化图表</p>
                </div>
            `;
        };

    } catch (error) {
        console.error('图像加载错误:', error);
        imageContainer.innerHTML = `
            <div class="image-error">
                <i class="fas fa-exclamation-circle"></i>
                <p>图表加载失败</p>
            </div>
        `;
    }
}

// 显示结果区域
function showResults() {
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('errorSection').style.display = 'none';

    // 平滑滚动到结果
    document.getElementById('resultsSection').scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// 显示错误
function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorSection').style.display = 'block';

    // 滚动到错误区域
    document.getElementById('errorSection').scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// 隐藏错误
function hideError() {
    document.getElementById('errorSection').style.display = 'none';
}

// 重置表单
function resetForm() {
    // 重置文件输入
    document.getElementById('reference_file').value = '';
    document.getElementById('test_file').value = '';

    // 重置文件信息显示
    updateFileInfo('reference_info', null);
    updateFileInfo('test_info', null);

    // 隐藏结果和错误
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';

    // 重置进度条
    hideProgress();

    // 滚动到顶部
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // 清除会话数据
    if (currentSessionId) {
        cleanupSession(currentSessionId);
        currentSessionId = null;
    }

    analysisResults = null;
}

// 清理会话文件
async function cleanupSession(sessionId) {
    try {
        await fetch(`/cleanup/${sessionId}`, {
            method: 'POST'
        });
    } catch (error) {
        console.error('清理会话失败:', error);
    }
}

// 下载报告
function downloadReport() {
    if (!analysisResults) {
        showError('没有可下载的报告');
        return;
    }

    // 创建报告文本
    const reportText = createReportText();

    // 创建Blob并下载
    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `midi_analysis_${currentSessionId || 'report'}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// 创建报告文本
function createReportText() {
    const { accuracy, rhythm } = analysisResults;
    const timestamp = new Date().toLocaleString('zh-CN');

    return `
MIDI评分系统 - 分析报告
================================
生成时间: ${timestamp}
会话ID: ${currentSessionId || 'N/A'}

音符准确度分析
----------------
精确率 (Precision): ${accuracy?.precision || 0}
召回率 (Recall): ${accuracy?.recall || 0}
F1分数: ${accuracy?.f1 || 0}
时间容差: ±${accuracy?.onset_tol || 0} 拍
音高容差: ±${accuracy?.pitch_tol || 0} 半音

节奏分析
----------------
DTW距离: ${rhythm?.dtw_distance || 0}
平均时间偏差: ${rhythm?.avg_time_deviation || 0} 拍
最大时间偏差: ${rhythm?.max_deviation || 0} 拍

可视化图表已保存为PNG图像。

备注
----------------
- 基于MIDI评分系统 v4.5
- 分析引擎: music21 + FastDTW
- 生成可视化: Matplotlib
    `.trim();
}

// 检查是否为MIDI文件
function isMidiFile(file) {
    if (!file) return false;
    const validExtensions = ['.mid', '.midi'];
    const fileName = file.name.toLowerCase();
    return validExtensions.some(ext => fileName.endsWith(ext));
}

// 初始化工具提示
function initTooltips() {
    // 可以在这里添加工具提示初始化代码
    console.log('工具提示已初始化');
}

// 键盘快捷键
document.addEventListener('keydown', function(e) {
    // Ctrl+Enter 开始分析
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        analyzeFiles();
    }

    // Esc 重置表单
    if (e.key === 'Escape') {
        resetForm();
    }
});