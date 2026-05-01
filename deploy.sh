#!/bin/bash
# ============================================
# MP3 → MIDI 转换服务 - 腾讯云 CVM 一键部署脚本
# 适用于 Ubuntu 20.04 / 22.04
# ============================================
set -e

echo "======================================"
echo "  MP3 → MIDI 转换服务 - 部署脚本"
echo "======================================"
echo ""

PROJECT_DIR="/opt/midi-converter"
APP_PORT=8080

# ---------- 1. 安装系统依赖 ----------
echo "[1/5] 安装系统依赖..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    nginx \
    supervisor

# ---------- 2. 创建项目目录 ----------
echo "[2/5] 设置项目目录..."
sudo mkdir -p "$PROJECT_DIR"
sudo cp -r web_app "$PROJECT_DIR/"
sudo cp -r src "$PROJECT_DIR/"
sudo cp web_app/requirements.txt "$PROJECT_DIR/"

# ---------- 3. 创建虚拟环境并安装依赖 ----------
echo "[3/5] 安装 Python 依赖..."
cd "$PROJECT_DIR"
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple

# ---------- 4. 配置 Nginx ----------
echo "[4/5] 配置 Nginx..."
sudo tee /etc/nginx/sites-available/midi-converter > /dev/null << 'NGINX'
server {
    listen 80;
    server_name _;

    client_max_body_size 50M;
    proxy_read_timeout 180s;
    proxy_send_timeout 180s;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
NGINX

sudo ln -sf /etc/nginx/sites-available/midi-converter /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# ---------- 5. 配置 Supervisor 守护进程 ----------
echo "[5/5] 配置 Supervisor 守护进程..."
sudo tee /etc/supervisor/conf.d/midi-converter.conf > /dev/null << 'SUPERVISOR'
[program:midi-converter]
directory=/opt/midi-converter/web_app
command=/opt/midi-converter/venv/bin/uvicorn app:app --host 127.0.0.1 --port 8080
autostart=true
autorestart=true
stderr_logfile=/var/log/midi-converter.err.log
stdout_logfile=/var/log/midi-converter.out.log
environment=PATH="/opt/midi-converter/venv/bin"
SUPERVISOR

sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart midi-converter

# ---------- 完成 ----------
echo ""
echo "======================================"
echo "  部署完成!"
echo "======================================"
echo ""
echo "  服务地址: http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_IP')"
echo "  服务端口: 80"
echo ""
echo "  管理命令:"
echo "    查看状态: sudo supervisorctl status"
echo "    重启服务: sudo supervisorctl restart midi-converter"
echo "    查看日志: sudo tail -f /var/log/midi-converter.out.log"
echo ""
