"""Generate examples/sample_paper.pdf and examples/sample_chart.png for the demo.

Uses only Pillow + PyMuPDF (already in requirements). No matplotlib dependency.
Run with: .venv\\Scripts\\python.exe scripts\\make_demo_assets.py
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
EXAMPLES.mkdir(exist_ok=True)


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_chart(
    width: int,
    height: int,
    title: str,
    points: list[tuple[str, float]],
    y_max: float,
    y_label: str,
    bar_color: tuple[int, int, int] = (72, 118, 196),
) -> Image.Image:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    title_f = _get_font(24)
    axis_f = _get_font(14)
    label_f = _get_font(13)

    margin_l, margin_r, margin_t, margin_b = 80, 40, 60, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    # Title
    draw.text((margin_l, 18), title, fill=(30, 30, 30), font=title_f)

    # Axes
    draw.line(
        [(margin_l, margin_t), (margin_l, margin_t + plot_h)], fill=(0, 0, 0), width=2
    )
    draw.line(
        [(margin_l, margin_t + plot_h), (margin_l + plot_w, margin_t + plot_h)],
        fill=(0, 0, 0),
        width=2,
    )

    # Y grid + labels
    for i in range(6):
        y = margin_t + plot_h - (plot_h * i / 5)
        draw.line(
            [(margin_l - 5, y), (margin_l + plot_w, y)],
            fill=(220, 220, 220),
            width=1,
        )
        val = y_max * i / 5
        draw.text((margin_l - 50, y - 8), f"{val:.0f}", fill=(60, 60, 60), font=axis_f)

    # Y-axis label (vertical text approximated by rotating a small image)
    y_label_img = Image.new("RGB", (len(y_label) * 9, 20), (255, 255, 255))
    ImageDraw.Draw(y_label_img).text((0, 2), y_label, fill=(30, 30, 30), font=axis_f)
    y_label_img = y_label_img.rotate(90, expand=True)
    img.paste(y_label_img, (10, margin_t + plot_h // 2 - y_label_img.height // 2))

    # Bars
    n = len(points)
    bar_w = plot_w / n * 0.6
    gap = plot_w / n * 0.4 / 2
    for i, (label, val) in enumerate(points):
        x0 = margin_l + i * (plot_w / n) + gap
        bh = plot_h * (val / y_max)
        y0 = margin_t + plot_h - bh
        draw.rectangle([x0, y0, x0 + bar_w, margin_t + plot_h], fill=bar_color)
        # Value on top of the bar
        draw.text((x0, y0 - 18), f"{val:.1f}", fill=(30, 30, 30), font=label_f)
        # X label below
        draw.text(
            (x0, margin_t + plot_h + 8),
            label,
            fill=(30, 30, 30),
            font=label_f,
        )

    return img


def make_paper_figure() -> Image.Image:
    """A results bar chart referenced inside the sample paper."""
    return _draw_chart(
        width=700,
        height=420,
        title="Figure 2: Top-1 accuracy across benchmarks",
        points=[
            ("CIFAR-10", 94.2),
            ("CIFAR-100", 76.8),
            ("ImageNet", 78.4),
            ("SVHN", 96.1),
        ],
        y_max=100.0,
        y_label="Top-1 accuracy (%)",
    )


def make_standalone_chart() -> Image.Image:
    """The separately-uploaded user image — a trend that *differs* from Figure 2."""
    return _draw_chart(
        width=700,
        height=420,
        title="Training loss over epochs (our run)",
        points=[
            ("ep1", 2.31),
            ("ep2", 1.74),
            ("ep3", 1.22),
            ("ep4", 0.95),
            ("ep5", 0.78),
            ("ep6", 0.65),
            ("ep7", 0.58),
            ("ep8", 0.54),
        ],
        y_max=2.5,
        y_label="Cross-entropy loss",
        bar_color=(196, 96, 72),
    )


PAPER_SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Abstract",
        [
            "We present Nebula-Net, a compact vision encoder that combines cross-modal "
            "attention with a lightweight convolutional trunk. Nebula-Net is trained on a "
            "mixture of natural images and synthetic augmentations, and evaluated on four "
            "standard benchmarks. On ImageNet the model reaches 78.4% top-1 accuracy with "
            "only 12.3M parameters, outperforming prior art at comparable compute.",
        ],
    ),
    (
        "1. Introduction",
        [
            "Efficient image classification remains a central problem in computer vision. "
            "While large transformers have set records on accuracy, their parameter counts "
            "and latency profiles make them impractical for on-device deployment. In this "
            "work we revisit small convolutional architectures and show that a careful "
            "pairing of depth-wise separable convolutions with lightweight cross-attention "
            "layers yields a strong accuracy/compute trade-off.",
            "Our contributions are threefold. First, we introduce Nebula-Net, a 12.3M-"
            "parameter backbone. Second, we propose a training recipe that combines "
            "RandAugment-style augmentations with a label-smoothed cross-entropy loss. "
            "Third, we provide extensive ablations across four benchmarks.",
        ],
    ),
    (
        "2. Method",
        [
            "Nebula-Net consists of five stages. Each stage halves the spatial resolution "
            "and doubles the channel count. Within each stage, inverted residual blocks "
            "are interleaved with a single cross-attention layer that exchanges "
            "information across spatial tokens. We use GELU activations throughout and "
            "apply dropout with probability 0.1 on the final classifier layer.",
            "We train for 120 epochs using AdamW with a base learning rate of 3e-4 and "
            "cosine decay. Batch size is 512. Mixed-precision training is used throughout.",
        ],
    ),
    (
        "3. Experiments",
        [
            "We evaluate Nebula-Net on CIFAR-10, CIFAR-100, ImageNet, and SVHN. All "
            "datasets are used at their standard resolutions. We report top-1 accuracy on "
            "the held-out validation split for each benchmark. The complete results are "
            "shown in Table 1 and visualized in Figure 2.",
            "Table 1 also reports the approximate inference latency measured on a single "
            "consumer GPU with batch size 1 at FP16 precision.",
        ],
    ),
    (
        "4. Discussion",
        [
            "The numbers indicate that Nebula-Net is most competitive on mid-resolution "
            "datasets like CIFAR-100 and ImageNet, where its 12.3M parameter budget still "
            "leaves room for strong feature diversity. The model is less dominant on SVHN "
            "where simpler baselines already saturate the benchmark.",
            "Training loss curves from our main ImageNet run are presented separately in "
            "the supplementary. The loss decreases smoothly without instabilities and we "
            "observe no divergence events across three random seeds.",
        ],
    ),
    (
        "5. Conclusion",
        [
            "We described Nebula-Net, a compact image encoder that achieves 78.4% top-1 "
            "accuracy on ImageNet with 12.3M parameters. We believe the accuracy/compute "
            "trade-off makes it a practical choice for on-device inference.",
        ],
    ),
]


def _insert_table(page: fitz.Page, y: float) -> float:
    """Draw a results table on the page. Returns new y-cursor below the table."""
    # Draw as lines + text so PyMuPDF/Unstructured can pick it up.
    col_x = [72, 220, 340, 460, 540]
    headers = ["Dataset", "Accuracy (%)", "Params (M)", "Latency (ms)", "Year"]
    rows = [
        ["CIFAR-10", "94.2", "12.3", "4.1", "2026"],
        ["CIFAR-100", "76.8", "12.3", "4.1", "2026"],
        ["ImageNet", "78.4", "12.3", "6.8", "2026"],
        ["SVHN", "96.1", "12.3", "3.9", "2026"],
    ]

    row_h = 20
    # Header
    page.draw_line((col_x[0], y), (col_x[-1] + 50, y), color=(0, 0, 0), width=0.8)
    for i, h in enumerate(headers):
        page.insert_text((col_x[i], y + 14), h, fontsize=10, fontname="helv")
    page.draw_line(
        (col_x[0], y + row_h), (col_x[-1] + 50, y + row_h), color=(0, 0, 0), width=0.6
    )

    y += row_h
    for row in rows:
        for i, cell in enumerate(row):
            page.insert_text((col_x[i], y + 14), cell, fontsize=10, fontname="helv")
        y += row_h
    page.draw_line(
        (col_x[0], y), (col_x[-1] + 50, y), color=(0, 0, 0), width=0.8
    )
    page.insert_text(
        (72, y + 18), "Table 1: Results on four benchmarks.", fontsize=9, fontname="helv"
    )
    return y + 34


def _wrap_text(text: str, width_chars: int = 92) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > width_chars and cur:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += len(w) + 1
    if cur:
        lines.append(" ".join(cur))
    return lines


def build_pdf() -> Path:
    pdf = fitz.open()
    page = pdf.new_page(width=612, height=792)  # US Letter
    y = 72

    # Title
    page.insert_text(
        (72, y), "Nebula-Net: Compact Vision Encoders Revisited", fontsize=18, fontname="helv"
    )
    y += 26
    page.insert_text((72, y), "Anonymous Authors  |  2026", fontsize=11, fontname="helv")
    y += 28

    # Body
    body_font_size = 10
    line_h = 13

    for heading, paragraphs in PAPER_SECTIONS:
        if y > 720:
            page = pdf.new_page(width=612, height=792)
            y = 72
        page.insert_text((72, y), heading, fontsize=12, fontname="helv")
        y += 18
        for para in paragraphs:
            for line in _wrap_text(para):
                if y > 760:
                    page = pdf.new_page(width=612, height=792)
                    y = 72
                page.insert_text((72, y), line, fontsize=body_font_size, fontname="helv")
                y += line_h
            y += 6  # paragraph spacing
        y += 10
        if heading.startswith("3."):
            # Insert the table right after section 3.
            if y > 640:
                page = pdf.new_page(width=612, height=792)
                y = 72
            y = _insert_table(page, y)

            # Insert Figure 2 on a new page for clarity.
            page = pdf.new_page(width=612, height=792)
            fig = make_paper_figure()
            buf = io.BytesIO()
            fig.save(buf, format="PNG")
            rect = fitz.Rect(72, 80, 540, 360)
            page.insert_image(rect, stream=buf.getvalue())
            page.insert_text(
                (72, 380),
                "Figure 2: Top-1 accuracy per benchmark. All values from Table 1.",
                fontsize=9,
                fontname="helv",
            )
            y = 410

    out_path = EXAMPLES / "sample_paper.pdf"
    pdf.save(str(out_path))
    pdf.close()
    return out_path


def build_chart() -> Path:
    img = make_standalone_chart()
    out_path = EXAMPLES / "sample_chart.png"
    img.save(out_path, format="PNG")
    return out_path


if __name__ == "__main__":
    pdf_path = build_pdf()
    chart_path = build_chart()
    print(f"Wrote {pdf_path}")
    print(f"Wrote {chart_path}")
