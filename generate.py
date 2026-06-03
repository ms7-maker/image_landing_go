"""
generate.py — CLI-утилита для генерации изображения landing page.

Как запустить (Windows / PowerShell):
1) (Рекомендуется) создайте и активируйте venv:
   py -m venv .venv
   .\\.venv\\Scripts\\Activate.ps1

2) Установите зависимости:
   pip install -r requirements.txt

3) Создайте .env:
   - Скопируйте .env.example в .env
   - Укажите ключ:
     OPENAI_KEY=ваш_ключ

4) Запуск:
   python generate.py "лендинг для онлайн-школы"
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from generate_image import generate_image
from openai_reasoning import enhance_prompt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate.py",
        description=(
            "Генерирует PNG (1024x1024) первой страницы landing page по текстовому запросу.\n"
            "Сначала улучшает промпт через gpt-4.1-mini, затем генерирует картинку через gpt-image-1."
        ),
    )
    parser.add_argument(
        "prompt",
        type=str,
        help='Текстовый запрос, например: "лендинг для онлайн-школы"',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        enhanced = enhance_prompt(args.prompt)
        image_path = generate_image(enhanced)
    except Exception as exc:  # noqa: BLE001 - CLI должен показывать понятную ошибку
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print("Изображение успешно создано!")
    print(f"Путь: {image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

