"""Extract readable blocks and embedded figures from a self-contained HTML report."""

from __future__ import annotations

import argparse
import base64
import json
import re
from html.parser import HTMLParser
from pathlib import Path


BLOCK_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "p",
    "li",
    "figcaption",
    "caption",
    "th",
    "td",
    "div",
}


class ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.records: list[dict[str, object]] = []
        self.collectors: list[dict[str, object]] = []
        self.skip_depth = 0
        self.images: list[dict[str, str]] = []
        self.svgs: list[str] = []
        self._svg_depth = 0
        self._svg_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        if tag in {"script", "style"}:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "img":
            self.images.append(attrs_dict)
        if tag == "svg":
            self._svg_depth = 1
            self._svg_parts = [self.get_starttag_text()]
        elif self._svg_depth:
            self._svg_depth += 1
            self._svg_parts.append(self.get_starttag_text())
        if tag in BLOCK_TAGS:
            self.collectors.append({"tag": tag, "attrs": attrs_dict, "chunks": []})

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.skip_depth:
            return
        if self._svg_depth:
            self._svg_parts.append(self.get_starttag_text())
        if tag == "img":
            self.images.append({key: value or "" for key, value in attrs})

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if self._svg_depth:
            self._svg_parts.append(f"</{tag}>")
            self._svg_depth -= 1
            if self._svg_depth == 0:
                self.svgs.append("".join(self._svg_parts))
                self._svg_parts = []
        for index in range(len(self.collectors) - 1, -1, -1):
            collector = self.collectors[index]
            if collector["tag"] == tag:
                self.collectors.pop(index)
                text = re.sub(r"\s+", " ", "".join(collector["chunks"])).strip()
                if text:
                    self.records.append(
                        {"tag": tag, "attrs": collector["attrs"], "text": text}
                    )
                break

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self._svg_depth:
            self._svg_parts.append(data)
        for collector in self.collectors:
            collector["chunks"].append(data)


def extract_images(images: list[dict[str, str]], output_dir: Path) -> list[dict[str, str]]:
    image_dir = output_dir / "embedded_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for index, attrs in enumerate(images, start=1):
        src = attrs.get("src", "")
        item = {"index": str(index), "alt": attrs.get("alt", ""), "src": src[:200]}
        match = re.fullmatch(r"data:image/([^;,]+);base64,(.*)", src, re.DOTALL)
        if match:
            extension = {"jpeg": "jpg", "svg+xml": "svg"}.get(match.group(1), match.group(1))
            path = image_dir / f"image_{index:03d}.{extension}"
            path.write_bytes(base64.b64decode(match.group(2)))
            item["file"] = str(path)
        manifest.append(item)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    html = args.html.read_text(encoding="utf-8", errors="replace")
    report = ReportParser()
    report.feed(html)
    args.output.mkdir(parents=True, exist_ok=True)

    images = extract_images(report.images, args.output)
    svg_dir = args.output / "inline_svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    for index, svg in enumerate(report.svgs, start=1):
        (svg_dir / f"figure_{index:03d}.svg").write_text(svg, encoding="utf-8")

    (args.output / "records.json").write_text(
        json.dumps(report.records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "images.json").write_text(
        json.dumps(images, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = []
    for record in report.records:
        tag = str(record["tag"])
        if tag != "div":
            lines.append(f"[{tag.upper()}] {record['text']}")
    (args.output / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "input": str(args.html),
        "records": len(report.records),
        "images": len(report.images),
        "svgs": len(report.svgs),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
