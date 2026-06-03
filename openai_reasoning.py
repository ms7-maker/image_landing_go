"""
openai_reasoning.py

Инструкция (быстрый старт):
1) Установите зависимости:
   pip install -r requirements.txt

2) Создайте .env:
   Скопируйте .env.example в .env и вставьте ключ:
   OPENAI_KEY=...

3) Использование:
   Этот модуль улучшает (детализирует) пользовательский промпт для генерации UI.
   Основная функция: enhance_prompt(user_prompt: str) -> str
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


def _get_openai_key() -> str:
    """
    Достаём API-ключ из окружения.
    По ТЗ используем OPENAI_KEY, но для совместимости также поддерживаем OPENAI_API_KEY.
    """

    load_dotenv()  # безопасно вызывать много раз

    key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Не найден API-ключ. Создайте файл .env и задайте OPENAI_KEY=... "
            "(или OPENAI_API_KEY=...)."
        )
    return key


def enhance_prompt(user_prompt: str) -> str:
    """
    Делает промпт более детализированным и «дизайнерским».

    Логика:
    - Берём исходный промпт пользователя (1 фраза)
    - Просим модель gpt-4.1-mini превратить его в подробное описание современного landing page UI
    - Возвращаем улучшенный промпт (только описание, без лишнего текста)
    """

    if not user_prompt or not user_prompt.strip():
        raise ValueError("Промпт пользователя пустой. Передайте непустую строку.")

    client = OpenAI(api_key=_get_openai_key())

    system_instructions = (
        "You are a helpful UI/UX prompt enhancer.\n"
        "Transform a short idea into a detailed image prompt for a modern landing page.\n"
        "Must include: modern landing page, clean UI, hero section, high-quality UI/UX design.\n"
        "Focus on layout and visual details: typography, spacing, grid, colors, components.\n"
        "Output ONLY the enhanced prompt (no preface, no bullet numbering, no extra commentary).\n"
        "All visible UI text must be in Russian (Cyrillic).\n"
        "Use natural, marketing-ready Russian copy (без транслита/английских слов).\n"
        "Include Russian headings, subheadings, button labels, menu items, pricing cards, etc.\n"

    )

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_prompt.strip()},
            ],
        )
    except Exception as exc:  # noqa: BLE001 - хотим дружелюбную ошибку для новичка
        raise RuntimeError(f"Ошибка OpenAI API при улучшении промпта: {exc}") from exc

    # У SDK есть удобное поле output_text (если оно доступно).
    enhanced = getattr(response, "output_text", None)
    if enhanced is None:
        # Фолбэк: пытаемся собрать текст из response.output (на случай изменений SDK).
        enhanced_parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    enhanced_parts.append(text)
        enhanced = "\n".join(enhanced_parts).strip()

    enhanced = (enhanced or "").strip()
    if not enhanced:
        raise RuntimeError("OpenAI вернул пустой улучшенный промпт. Попробуйте ещё раз.")

    return enhanced

