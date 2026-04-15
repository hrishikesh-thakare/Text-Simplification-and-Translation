"""Runtime-only desktop launcher for the local Streamlit app.

This keeps the app focused on the production runtime path:
base model + LoRA adapter + Indic translation.
"""

from __future__ import annotations

import os
import socket
import time
import threading
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
APP_SCRIPT = PROJECT_ROOT / "runtime" / "streamlit_app.py"
HOST = os.getenv("APP_HOST", "127.0.0.1")
PORT = int(os.getenv("APP_PORT", "8501"))


def _wait_for_port(host: str, port: int, timeout_seconds: int = 30) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def main() -> None:
    if not APP_SCRIPT.exists():
        raise FileNotFoundError(f"Runtime app script not found: {APP_SCRIPT}")

    from streamlit import config as st_config
    from streamlit.web import bootstrap

    # Mirror Streamlit CLI config setup so static assets and routes resolve correctly.
    st_config._main_script_path = str(APP_SCRIPT)
    flag_options = {
        "global_developmentMode": False,
        "server_address": HOST,
        "server_port": PORT,
        "server_headless": True,
    }
    bootstrap.load_config_options(flag_options)

    def open_browser_when_ready() -> None:
        if _wait_for_port(HOST, PORT):
            webbrowser.open_new(f"http://{HOST}:{PORT}")

    threading.Thread(target=open_browser_when_ready, daemon=True).start()
    bootstrap.run(str(APP_SCRIPT), False, [], flag_options)


if __name__ == "__main__":
    main()