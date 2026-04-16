"""
MIDI Score Evaluation Web Application
基于优化的MIDI评分系统，提供Web界面
"""
import os
import uuid
import shutil
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename

# 导入分析模块
import sys
sys.path.append('..')  # 添加父目录以导入midi_evaluator.py
from midi_evaluator import ScoreEvaluator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB最大文件大小
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ANALYSIS_FOLDER'] = 'analysis_outputs'
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANALYSIS_FOLDER'], exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'mid', 'midi'}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """主页 - 文件上传表单"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """处理文件上传并进行分析"""
    try:
        # 检查是否有文件被上传
        if 'reference_file' not in request.files or 'test_file' not in request.files:
            return jsonify({'error': '请选择参考文件和测试文件'}), 400

        reference_file = request.files['reference_file']
        test_file = request.files['test_file']

        # 检查文件名
        if reference_file.filename == '' or test_file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400

        if not (allowed_file(reference_file.filename) and allowed_file(test_file.filename)):
            return jsonify({'error': '只支持MIDI文件 (.mid, .midi)'}), 400

        # 生成唯一会话ID
        session_id = str(uuid.uuid4())[:8]
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)

        # 保存文件
        ref_filename = secure_filename(f"reference_{session_id}.mid")
        test_filename = secure_filename(f"test_{session_id}.mid")
        ref_path = os.path.join(session_folder, ref_filename)
        test_path = os.path.join(session_folder, test_filename)

        reference_file.save(ref_path)
        test_file.save(test_path)

        # 调用分析函数
        results = analyze_midi(ref_path, test_path, session_id)

        # 返回结果
        return jsonify(results)

    except Exception as e:
        app.logger.error(f"分析错误: {str(e)}")
        return jsonify({'error': f'分析失败: {str(e)}'}), 500

def analyze_midi(ref_path, test_path, session_id):
    """分析MIDI文件并返回结果"""
    try:
        # 创建分析器实例
        evaluator = ScoreEvaluator(Path(ref_path), Path(test_path))

        # 获取分析结果
        accuracy = evaluator.note_accuracy()
        rhythm = evaluator.rhythm_analysis()

        # 生成可视化报告
        output_folder = os.path.join(app.config['ANALYSIS_FOLDER'], session_id)
        os.makedirs(output_folder, exist_ok=True)

        # 生成可视化（直接保存到指定文件夹）
        image_path = evaluator.generate_visual_report(output_dir=output_folder)

        # 获取图像文件名
        image_filename = os.path.basename(image_path)

        # 准备结果
        results = {
            'session_id': session_id,
            'accuracy': accuracy,
            'rhythm': rhythm,
            'image_url': f'/analysis_output/{session_id}/{image_filename}'
        }

        return results

    except Exception as e:
        raise Exception(f"MIDI分析错误: {str(e)}")

@app.route('/analysis_output/<session_id>/<filename>')
def serve_analysis_output(session_id, filename):
    """提供分析结果图像"""
    folder_path = os.path.join(app.config['ANALYSIS_FOLDER'], session_id)
    return send_from_directory(folder_path, filename)

@app.route('/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    """清理会话文件（可选）"""
    try:
        upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        analysis_folder = os.path.join(app.config['ANALYSIS_FOLDER'], session_id)

        if os.path.exists(upload_folder):
            shutil.rmtree(upload_folder)
        if os.path.exists(analysis_folder):
            shutil.rmtree(analysis_folder)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)