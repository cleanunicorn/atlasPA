PROJECT_DIR := $(shell pwd)
PYTHON      := uv run python
SERVICE     := atlas
SERVICE_FILE := $(HOME)/.config/systemd/user/$(SERVICE).service

.PHONY: help install setup run cli discord web logs test \
        service-install service-uninstall service-start service-stop \
        service-restart service-status service-logs

help:
	@echo "Atlas — Personal AI Agent"
	@echo ""
	@echo "Usage:"
	@echo "  make install          Install dependencies"
	@echo "  make setup            Interactive setup wizard (create config/.env)"
	@echo ""
	@echo "  make run              Start Telegram bot (default)"
	@echo "  make cli              Interactive terminal session"
	@echo "  make discord          Discord bot"
	@echo "  make web              Browser-based web UI (http://localhost:7860)"
	@echo "  make logs             Browse LLM logs in the browser"
	@echo "  make test             Run test suite"
	@echo ""
	@echo "  make service-install  Install systemd user service"
	@echo "  make service-uninstall Remove systemd user service"
	@echo "  make service-start    Start the service"
	@echo "  make service-stop     Stop the service"
	@echo "  make service-restart  Restart the service"
	@echo "  make service-status   Show service status"
	@echo "  make service-logs     Follow service logs"

install:
	uv sync

setup:
	$(PYTHON) main.py setup

run:
	$(PYTHON) main.py run

cli:
	$(PYTHON) main.py run --cli

discord:
	$(PYTHON) main.py run --discord

web:
	$(PYTHON) main.py run --web

logs:
	$(PYTHON) main.py logs

test:
	uv run python -m pytest tests/ -v

# ── systemd user service ───────────────────────────────────────────────────────

service-install:
	@mkdir -p $(HOME)/.config/systemd/user
	@echo "[Unit]" > $(SERVICE_FILE)
	@echo "Description=Atlas Personal AI Agent" >> $(SERVICE_FILE)
	@echo "After=network-online.target" >> $(SERVICE_FILE)
	@echo "Wants=network-online.target" >> $(SERVICE_FILE)
	@echo "" >> $(SERVICE_FILE)
	@echo "[Service]" >> $(SERVICE_FILE)
	@echo "Type=simple" >> $(SERVICE_FILE)
	@echo "WorkingDirectory=$(PROJECT_DIR)" >> $(SERVICE_FILE)
	@echo "ExecStart=$(shell which uv) run python main.py run --skip-checks" >> $(SERVICE_FILE)
	@echo "Restart=on-failure" >> $(SERVICE_FILE)
	@echo "RestartSec=10" >> $(SERVICE_FILE)
	@echo "StandardOutput=journal" >> $(SERVICE_FILE)
	@echo "StandardError=journal" >> $(SERVICE_FILE)
	@echo "" >> $(SERVICE_FILE)
	@echo "[Install]" >> $(SERVICE_FILE)
	@echo "WantedBy=default.target" >> $(SERVICE_FILE)
	systemctl --user daemon-reload
	systemctl --user enable $(SERVICE)
	@echo "Service installed. Run 'make service-start' to start it."
	@echo "To persist across logout: loginctl enable-linger $(USER)"

service-uninstall:
	-systemctl --user stop $(SERVICE)
	-systemctl --user disable $(SERVICE)
	rm -f $(SERVICE_FILE)
	systemctl --user daemon-reload
	@echo "Service removed."

service-start:
	systemctl --user start $(SERVICE)

service-stop:
	systemctl --user stop $(SERVICE)

service-restart:
	systemctl --user restart $(SERVICE)

service-status:
	systemctl --user status $(SERVICE)

service-logs:
	journalctl --user -u $(SERVICE) -f
