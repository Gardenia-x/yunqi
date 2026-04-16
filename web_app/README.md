# MIDI评分系统 - Web版

基于优化的MIDI评分系统v4.5，提供完整的Web界面，允许用户上传两个MIDI文件并在线获取详细的分析报告。

## 功能特点

- **文件上传**: 支持上传参考MIDI文件和测试MIDI文件
- **实时分析**: 在服务器端进行MIDI解析和性能分析
- **可视化报告**: 生成包含节奏对齐、音高-时间对比、力度分析的可视化图表
- **详细指标**: 提供音符准确度（精确率、召回率、F1分数）和节奏分析（DTW距离、时间偏差）
- **响应式设计**: 适配桌面和移动设备
- **一键下载**: 支持下载完整的文本分析报告

## 系统架构

### 后端 (Flask + Python)
- **Flask应用**: 提供REST API和静态文件服务
- **MIDI分析引擎**: 基于`music21`、`fastdtw`、`numpy`的优化分析模块
- **文件管理**: 临时文件上传和会话隔离
- **图像生成**: 使用`matplotlib`生成可视化图表

### 前端 (HTML + CSS + JavaScript)
- **现代化UI**: 使用CSS Grid和Flexbox布局
- **交互式表单**: 文件拖放支持（未来扩展）
- **实时进度**: 分析过程进度条显示
- **动态更新**: AJAX数据加载和无刷新结果展示

## 安装和运行

### 前置要求
- Python 3.8+
- 已安装MIDI评分系统的所有依赖

### 1. 安装Web依赖
```bash
cd web_app
pip install -r requirements.txt
```

### 2. 运行Web应用
```bash
python app.py
```

应用将在 http://localhost:5000 启动

### 3. 使用浏览器访问
打开浏览器，访问 http://localhost:5000

## 使用说明

1. **上传文件**
   - 选择参考MIDI文件（标准版本）
   - 选择测试MIDI文件（待评估版本）

2. **开始分析**
   - 点击"开始分析"按钮
   - 等待分析完成（进度条显示）

3. **查看结果**
   - 查看音符准确度指标
   - 查看节奏分析结果
   - 查看可视化图表

4. **下载报告**
   - 点击"下载完整报告"获取文本格式分析结果

## 项目结构

```
web_app/
├── app.py                 # Flask主应用
├── requirements.txt       # Python依赖
├── README.md             # 说明文档
├── static/               # 静态资源
│   ├── style.css         # 样式表
│   └── script.js         # 前端交互脚本
├── templates/            # HTML模板
│   └── index.html        # 主页面
├── uploads/              # 上传文件存储（自动创建）
└── analysis_outputs/     # 分析结果图像（自动创建）
```

## API接口

### `POST /upload`
- **功能**: 上传并分析MIDI文件
- **参数**:
  - `reference_file`: 参考MIDI文件
  - `test_file`: 测试MIDI文件
- **返回**: JSON格式分析结果

### `GET /analysis_output/<session_id>/<filename>`
- **功能**: 获取分析生成的图像

### `POST /cleanup/<session_id>`
- **功能**: 清理会话临时文件

## 配置选项

在`app.py`中可以修改以下配置：

- `MAX_CONTENT_LENGTH`: 最大文件上传大小（默认16MB）
- `UPLOAD_FOLDER`: 上传文件存储目录
- `ANALYSIS_FOLDER`: 分析结果存储目录
- `SECRET_KEY`: Flask会话密钥（生产环境需修改）

## 故障排除

### 1. 导入错误
确保`midi_evaluator.py`在父目录中，且所有依赖已安装。

### 2. 文件上传失败
- 检查文件大小是否超过限制
- 确认文件格式为`.mid`或`.midi`
- 查看服务器日志获取详细错误信息

### 3. 分析过程缓慢
- 大型MIDI文件可能需要较长时间
- 考虑增加服务器资源

### 4. 图像无法显示
- 检查`analysis_outputs`目录权限
- 确认matplotlib后端配置正确

## 扩展计划

- [ ] 添加用户账户系统
- [ ] 支持批量文件分析
- [ ] 添加更多可视化图表类型
- [ ] 实现实时分析进度WebSocket
- [ ] 添加MIDI文件预览播放
- [ ] 支持导出PDF报告

## 技术栈

- **后端**: Flask, Werkzeug
- **分析引擎**: music21, fastdtw, numpy, scipy
- **可视化**: matplotlib, Pillow
- **前端**: HTML5, CSS3, JavaScript (ES6)
- **构建**: 纯Python，无需构建工具

## 许可证

本项目基于优化的MIDI评分系统，仅供学习和研究使用。

## 支持

如有问题或建议，请检查项目结构或查阅原始MIDI评分系统文档。