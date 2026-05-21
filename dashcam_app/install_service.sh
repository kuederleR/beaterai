#!/bin/bash

# Ensure script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (e.g. sudo ./install_service.sh)"
  exit 1
fi

# Get the absolute directory of where this script is located
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVICE_FILE="/etc/systemd/system/dashcam.service"

echo "Installing Dashcam service pointing to $APP_DIR..."

cat <<EOF > $SERVICE_FILE
[Unit]
Description=Dashcam Web Application Docker Compose Service
Requires=docker.service
After=docker.service network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$APP_DIR
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd, enable, and start the service
systemctl daemon-reload
systemctl enable dashcam.service
systemctl start dashcam.service

echo "============================================================"
echo "Dashcam service successfully installed and started!"
echo "It will now automatically boot up alongside your Jetson Orin."
echo "You can check its status anytime with: sudo systemctl status dashcam"
echo "============================================================"
