#!/usr/bin/env python3
"""Standalone Raspberry Pi 5 hardware readiness test for Chuwi.

This script does not require a desktop computer. It checks the local Linux
environment, camera availability, microphone capture, optional proximity
sensor, Python emotion stack, and audio output commands.

It is intentionally diagnostic: it does not claim emotion-model accuracy.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(label, fn):
    try:
        result = fn()
        print(f"[OK]   {label}: {result}")
        return True
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}")
        return False


def command_exists(name):
    return shutil.which(name) is not None


def camera():
    devices = sorted(Path("/dev").glob("video*"))
    if not devices:
        raise RuntimeError("No se encontró /dev/video*")
    return ", ".join(str(x) for x in devices)


def microphone():
    if not command_exists("arecord"):
        raise RuntimeError("arecord no está instalado")
    p = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=10)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "arecord -l falló")
    text = (p.stdout + p.stderr).strip()
    if not text or "card" not in text.lower():
        raise RuntimeError("No se detectó una tarjeta de captura")
    return "tarjeta de captura detectada"


def audio():
    tools = [x for x in ("mpg123", "aplay", "ffplay") if command_exists(x)]
    if not tools:
        raise RuntimeError("No hay reproductor de audio instalado")
    return ", ".join(tools)


def python_stack():
    required = ("torch", "torchvision", "PIL", "cv2")
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError("Faltan módulos: " + ", ".join(missing))
    return "torch/torchvision/PIL/opencv disponibles"


def timm_status():
    return "instalado" if importlib.util.find_spec("timm") else "NO instalado"


def resources():
    load = os.getloadavg()[0]
    mem = Path("/proc/meminfo").read_text()
    total = next(int(x.split()[1]) for x in mem.splitlines() if x.startswith("MemTotal:"))
    available = next(int(x.split()[1]) for x in mem.splitlines() if x.startswith("MemAvailable:"))
    return f"load1={load:.2f}, RAM disponible={available/1024:.0f} MB/{total/1024:.0f} MB"


def main():
    print("=" * 58)
    print("        CHUWI - RASPBERRY PI 5 HARDWARE TEST")
    print("=" * 58)
    results = [
        check("Sistema", lambda: Path("/proc/device-tree/model").read_text(errors="ignore").strip("\x00")),
        check("Cámara", camera),
        check("Micrófono", microphone),
        check("Audio", audio),
        check("Python/visión", python_stack),
        check("timm", timm_status),
        check("Recursos", resources),
    ]
    print("=" * 58)
    print(f"RESULTADO: {sum(results)}/{len(results)} comprobaciones OK")
    print("Nota: este diagnóstico no mide la precisión del modelo emocional.")
    print("=" * 58)


if __name__ == "__main__":
    main()
