"""
app.py — веб-интерфейс (Flask) для генерации изображения landing page.

Быстрый старт:
1) (Рекомендуется) активируйте venv:
   .\.venv\Scripts\Activate.ps1

2) Установите зависимости:
   pip install -r requirements.txt

3) Создайте .env (в корне проекта):
   OPENAI_KEY=ваш_ключ

4) Запустите сайт:
   python app.py

5) Откройте в браузере:
   http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for

from generate_image import generate_image, generate_image_via_proxy
from openai_reasoning import enhance_prompt


OUTPUTS_DIR = Path("outputs")


STYLE_OPTIONS: Dict[str, str] = {
    "modern": "modern",
    "minimalist": "minimalist",
    "professional": "professional",
    "uiux": "UI/UX design",
    "gradient": "gradient style",
    "futuristic": "futuristic",
    "cyberpunk": "cyberpunk",
    "cartoon": "cartoon",
    "sketch": "sketch",
}


@dataclass
class Job:
    id: str
    created_at: float
    prompt: str
    style: str
    progress: int = 0  # 0..100
    status: str = "queued"  # queued|running|done|error
    openai_image_path: Optional[str] = None
    proxy_image_path: Optional[str] = None
    openai_seconds: Optional[float] = None
    proxy_seconds: Optional[float] = None
    openai_error: Optional[str] = None
    proxy_error: Optional[str] = None
    error: Optional[str] = None  # общий текст ошибки (если обе генерации упали)


app = Flask(__name__)
# Для разработки удобнее видеть правки HTML сразу (без перезапуска).
# Это не влияет на логику генерации; просто отключает кэш шаблонов.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

_jobs: Dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _apply_style(user_prompt: str, style_key: str) -> str:
    """Добавляем стиль в запрос простым и прозрачным способом."""
    style_text = STYLE_OPTIONS.get(style_key, "")
    base = user_prompt.strip()
    if style_text:
        return f"{base}. Style: {style_text}."
    return base


def _run_job(job_id: str) -> None:
    """Фоновая задача: улучшить промпт и сгенерировать изображение."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job.status = "running"
        job.progress = 5

    try:
        # Шаг 1: улучшение промпта
        with _jobs_lock:
            job.progress = 15

        styled_prompt = _apply_style(job.prompt, job.style)
        enhanced = enhance_prompt(styled_prompt)
        # Дублируем constraint прямо перед Images API: так модель для изображений
        # получает явное требование к языку текста, даже если в улучшенном промпте
        # это сформулировано недостаточно жёстко.
        enhanced_for_images = (
            f"{enhanced}\n\n"
            "CRITICAL: All visible UI text must be in Russian (Cyrillic). "
            "No English words. No transliteration. Use natural marketing-ready Russian copy."
        )

        with _jobs_lock:
            job.progress = 55

        # Шаг 2: двойная генерация изображения (параллельно)
        with _jobs_lock:
            job.progress = 60

        results: dict[str, object] = {}

        def _run_openai() -> None:
            start = time.perf_counter()
            try:
                path = generate_image(enhanced_for_images)
                results["openai_path"] = path
            except Exception as exc:  # noqa: BLE001
                results["openai_error"] = str(exc)
            finally:
                results["openai_seconds"] = time.perf_counter() - start

        def _run_proxy() -> None:
            start = time.perf_counter()
            try:
                path = generate_image_via_proxy(enhanced_for_images)
                results["proxy_path"] = path
            except Exception as exc:  # noqa: BLE001
                results["proxy_error"] = str(exc)
            finally:
                results["proxy_seconds"] = time.perf_counter() - start

        t1 = threading.Thread(target=_run_openai, daemon=True)
        t2 = threading.Thread(target=_run_proxy, daemon=True)
        t1.start()
        t2.start()

        with _jobs_lock:
            job.progress = 70

        t1.join()
        with _jobs_lock:
            job.progress = 85

        t2.join()

        with _jobs_lock:
            job.progress = 100
            job.openai_image_path = results.get("openai_path") if isinstance(results.get("openai_path"), str) else None
            job.proxy_image_path = results.get("proxy_path") if isinstance(results.get("proxy_path"), str) else None
            job.openai_seconds = results.get("openai_seconds") if isinstance(results.get("openai_seconds"), float) else None
            job.proxy_seconds = results.get("proxy_seconds") if isinstance(results.get("proxy_seconds"), float) else None
            job.openai_error = results.get("openai_error") if isinstance(results.get("openai_error"), str) else None
            job.proxy_error = results.get("proxy_error") if isinstance(results.get("proxy_error"), str) else None

            # Если обе генерации упали — считаем задачу ошибкой
            if not job.openai_image_path and not job.proxy_image_path:
                job.status = "error"
                job.error = "Не удалось сгенерировать изображение ни напрямую, ни через ProxyAPI."
            else:
                job.status = "done"
    except Exception as exc:  # noqa: BLE001
        with _jobs_lock:
            job.status = "error"
            job.error = str(exc)
            job.progress = 100


@app.get("/")
def index():
    return render_template("index.html", styles=STYLE_OPTIONS)


@app.post("/generate")
def generate():
    prompt = (request.form.get("prompt") or "").strip()
    style = (request.form.get("style") or "modern").strip()

    if not prompt:
        return render_template(
            "index.html",
            styles=STYLE_OPTIONS,
            error="Введите промпт (описание лендинга).",
            default_prompt="",
            default_style=style,
        )

    if style not in STYLE_OPTIONS:
        style = "modern"

    job_id = uuid.uuid4().hex
    job = Job(id=job_id, created_at=time.time(), prompt=prompt, style=style)

    with _jobs_lock:
        _jobs[job_id] = job

    t = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    t.start()

    return redirect(url_for("job_page", job_id=job_id))


@app.get("/job/<job_id>")
def job_page(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return render_template("job.html", not_found=True)

        openai_image_url = None
        proxy_image_url = None
        if job.status in ("done", "error"):
            if job.openai_image_path:
                openai_image_url = url_for("outputs_file", filename=Path(job.openai_image_path).name)
            if job.proxy_image_path:
                proxy_image_url = url_for("outputs_file", filename=Path(job.proxy_image_path).name)

        return render_template(
            "job.html",
            job_id=job.id,
            prompt=job.prompt,
            style=job.style,
            progress=job.progress,
            status=job.status,
            error=job.error,
            openai_image_url=openai_image_url,
            proxy_image_url=proxy_image_url,
            openai_seconds=job.openai_seconds,
            proxy_seconds=job.proxy_seconds,
            openai_error=job.openai_error,
            proxy_error=job.proxy_error,
            styles=STYLE_OPTIONS,
        )


@app.get("/status/<job_id>")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "not_found"}), 404

        openai_image_url = None
        proxy_image_url = None
        if job.status in ("done", "error"):
            if job.openai_image_path:
                openai_image_url = url_for("outputs_file", filename=Path(job.openai_image_path).name)
            if job.proxy_image_path:
                proxy_image_url = url_for("outputs_file", filename=Path(job.proxy_image_path).name)

        return jsonify(
            {
                "ok": True,
                "id": job.id,
                "status": job.status,
                "progress": job.progress,
                "error": job.error,
                "openai_image_url": openai_image_url,
                "proxy_image_url": proxy_image_url,
                "openai_seconds": job.openai_seconds,
                "proxy_seconds": job.proxy_seconds,
                "openai_error": job.openai_error,
                "proxy_error": job.proxy_error,
            }
        )


@app.get("/outputs/<path:filename>")
def outputs_file(filename: str):
    # Безопасная отдача файлов только из outputs/
    return send_from_directory(OUTPUTS_DIR, filename)


def _ensure_outputs_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    load_dotenv()
    _ensure_outputs_dir()

    # HOST=0.0.0.0 удобно в Docker/WSL, но по умолчанию оставим локально.
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)

