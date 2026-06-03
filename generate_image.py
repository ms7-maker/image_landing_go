"""
generate_image.py

Инструкция (быстрый старт):
1) Установите зависимости:
   pip install -r requirements.txt

2) Создайте .env:
   Скопируйте .env.example в .env и вставьте ключ:
   OPENAI_KEY=...

3) Использование:
   Этот модуль генерирует PNG-изображение по промпту через OpenAI Images API.
   Основная функция: generate_image(prompt: str) -> str
"""

from __future__ import annotations

import base64
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI


def _get_openai_key() -> str:
    """Читаем ключ из .env / окружения."""

    load_dotenv()

    key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Не найден API-ключ. Создайте файл .env и задайте OPENAI_KEY=... "
            "(или OPENAI_API_KEY=...)."
        )
    return key


def _get_proxyapi_key() -> str:
    """Читаем ключ ProxyAPI из .env / окружения."""

    load_dotenv()

    key = os.getenv("PROXYAPI_KEY")
    if not key:
        raise RuntimeError(
            "Не найден ключ PROXYAPI_KEY. Добавьте его в файл .env: PROXYAPI_KEY=..."
        )
    return key


def _generate_image_bytes(
    *,
    prompt: str,
    api_key: str,
    base_url: str | None,
    model: str,
    size: str | None,
) -> bytes:
    """
    Генерирует изображение и возвращает байты PNG.
    base_url=None => обычный OpenAI endpoint.
    """

    client = OpenAI(api_key=api_key, base_url=base_url)

    kwargs: dict[str, object] = {
        "model": model,
        "prompt": prompt.strip(),
    }
    if size:
        kwargs["size"] = size

    try:
        result = client.images.generate(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Ошибка OpenAI Images API при генерации изображения: {exc}") from exc

    # Ожидаем base64 PNG в result.data[0].b64_json
    try:
        b64_data = result.data[0].b64_json
        if not b64_data:
            raise ValueError("Пустые данные изображения (b64_json).")
        return base64.b64decode(b64_data)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Не удалось прочитать данные изображения из ответа API: {exc}") from exc


def _save_image_bytes(*, image_bytes: bytes, filename_prefix: str) -> str:
    out_dir = Path("outputs")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Не удалось создать папку outputs/: {exc}") from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nonce = uuid4().hex[:8]
    out_path = out_dir / f"{filename_prefix}_{timestamp}_{nonce}.png"

    try:
        out_path.write_bytes(image_bytes)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Не удалось сохранить файл изображения: {exc}") from exc

    return str(out_path)


def generate_image(prompt: str) -> str:
    """
    Генерирует изображение landing page (PNG 1024x1024) и сохраняет локально.

    - Папка: outputs/ (создаётся, если нет)
    - Имя файла: landing_<timestamp>.png
    - Возвращает путь к сохранённому файлу (строкой)
    """

    if not prompt or not prompt.strip():
        raise ValueError("Промпт пустой. Передайте непустую строку.")

    image_bytes = _generate_image_bytes(
        prompt=prompt,
        api_key=_get_openai_key(),
        base_url=None,
        model="gpt-image-1",
        size="1024x1024",
    )
    return _save_image_bytes(image_bytes=image_bytes, filename_prefix="landing_openai")


def generate_image_via_proxy(prompt: str) -> str:
    """
    Генерация изображения через ProxyAPI.

    - base_url: https://api.proxyapi.ru/openai/v1
    - ключ: PROXYAPI_KEY
    - модель: gpt-image-2 (как в примере)
    """

    if not prompt or not prompt.strip():
        raise ValueError("Промпт пустой. Передайте непустую строку.")

    image_bytes = _generate_image_bytes(
        prompt=prompt,
        api_key=_get_proxyapi_key(),
        base_url="https://api.proxyapi.ru/openai/v1",
        model="gpt-image-2",
        size=None,  # в примере size не передаётся
    )
    return _save_image_bytes(image_bytes=image_bytes, filename_prefix="landing_proxyapi")

