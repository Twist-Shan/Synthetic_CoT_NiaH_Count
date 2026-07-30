from pathlib import Path

from PIL import Image, ImageDraw


root = Path(__file__).parent / "pdfs" / "final_render"
pages = sorted(root.glob("page-*.png"))
for group_start in range(0, len(pages), 5):
    group = pages[group_start : group_start + 5]
    sheet = Image.new("RGB", (1200, 1950), "white")
    draw = ImageDraw.Draw(sheet)
    for offset, path in enumerate(group):
        image = Image.open(path).convert("RGB")
        image.thumbnail((570, 600))
        col, row = offset % 2, offset // 2
        x, y = 15 + col * 595, 25 + row * 640
        sheet.paste(image, (x, y))
        draw.text((x, y - 18), path.stem, fill="black")
    first_page = group_start + 1
    last_page = group_start + len(group)
    sheet.save(root / f"contact-{first_page:02d}-{last_page:02d}.png")
