#!/usr/bin/env python3
"""Build contact sheets for a visual audit of every image embedded in the report."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def build_contact_sheets(run_dir: Path, report: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    document = report.read_text(encoding="utf-8")
    matches = list(
        re.finditer(
            r'<img\s+src="data:image/png;base64,([^"]+)"\s+alt="([^"]*)"[^>]*>',
            document,
            flags=re.S,
        )
    )

    known: dict[str, str] = {}
    for path in run_dir.rglob("*.png"):
        if output_dir in path.parents:
            continue
        try:
            known[_sha256(path.read_bytes())] = str(path.relative_to(run_dir)).replace("\\", "/")
        except OSError:
            continue

    records: list[dict[str, object]] = []
    images: list[Image.Image] = []
    for index, match in enumerate(matches, start=1):
        raw = base64.b64decode(match.group(1))
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        source = known.get(_sha256(raw), "embedded-only")
        records.append(
            {
                "index": index,
                "source": source,
                "width": image.width,
                "height": image.height,
                "aspect_ratio": round(image.width / image.height, 4),
                "alt": html.unescape(match.group(2)),
            }
        )
        images.append(image)

    manifest_path = output_dir / "embedded_figure_audit.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    cell_width, cell_height = 720, 500
    chart_width, chart_height = 680, 405
    per_sheet = 6
    sheets: list[Path] = []
    title_font = _font(19)
    detail_font = _font(15)
    for start in range(0, len(images), per_sheet):
        page = Image.new("RGB", (cell_width * 2, cell_height * 3), "white")
        draw = ImageDraw.Draw(page)
        for slot, (image, record) in enumerate(
            zip(images[start : start + per_sheet], records[start : start + per_sheet], strict=True)
        ):
            column, row = slot % 2, slot // 2
            x0, y0 = column * cell_width, row * cell_height
            thumb = image.copy()
            thumb.thumbnail((chart_width, chart_height), Image.Resampling.LANCZOS)
            chart_x = x0 + (cell_width - thumb.width) // 2
            chart_y = y0 + 68 + (chart_height - thumb.height) // 2
            page.paste(thumb, (chart_x, chart_y))
            draw.rectangle((x0, y0, x0 + cell_width - 1, y0 + cell_height - 1), outline="#cbd5e1", width=2)
            draw.text((x0 + 16, y0 + 12), f"{record['index']:02d}  {record['source']}", fill="#172033", font=title_font)
            draw.text(
                (x0 + 16, y0 + 40),
                f"{record['width']} x {record['height']}  ratio={record['aspect_ratio']}",
                fill="#526078",
                font=detail_font,
            )
        sheet_path = output_dir / f"contact_sheet_{start // per_sheet + 1:02d}.png"
        page.save(sheet_path, optimize=True)
        sheets.append(sheet_path)
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report = (args.report or run_dir / "v16_2_full_causal_report.html").resolve()
    output_dir = (args.output_dir or run_dir / "analysis" / "v10_port" / "qa").resolve()
    for path in build_contact_sheets(run_dir, report, output_dir):
        print(path)


if __name__ == "__main__":
    main()
