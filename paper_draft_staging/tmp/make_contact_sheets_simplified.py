from pathlib import Path
import sys

from PIL import Image, ImageDraw


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "qa_render_simplified"
pages = sorted(root.glob("page-*.png"))
for group_start in range(0, len(pages), 4):
    group = pages[group_start : group_start + 4]
    sheet = Image.new("RGB", (1400, 1900), "white")
    draw = ImageDraw.Draw(sheet)
    for offset, path in enumerate(group):
        page = Image.open(path).convert("RGB")
        page.thumbnail((670, 880))
        col, row = offset % 2, offset // 2
        x, y = 15 + col * 695, 35 + row * 930
        sheet.paste(page, (x, y))
        draw.text((x, 12 + row * 930), path.stem, fill="black")
    first_page = group_start + 1
    last_page = group_start + len(group)
    sheet.save(root / f"contact-{first_page:02d}-{last_page:02d}.png")
