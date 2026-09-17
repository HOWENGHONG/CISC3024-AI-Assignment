"""
Build the assignment report (Word .docx) from the experiment results.

    python report/make_report.py

Reads results/*.json and report/figures/*.png and writes
report/CISC3024_AI_Assignment1_HoWengHong_uc325107.docx
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "CISC3024_AI_Assignment1_HoWengHong_uc325107.docx")

sys.path.insert(0, os.path.join(ROOT, "src"))
from data import CIFAR10_CLASSES  # noqa: E402
from models import build_model, count_macs  # noqa: E402
from plot_results import ORDER, PRETTY, load_results  # noqa: E402

REPO_URL = "https://github.com/HOWENGHONG/CISC3024-AI-Assignment"
STUDENT = "Ho Weng Hong"
SID = "uc325107"


# --------------------------------------------------------------------------- #
# small python-docx helpers
# --------------------------------------------------------------------------- #
def set_cell_shading(cell, hex_fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def add_hyperlink(paragraph, url: str, text: str | None = None):
    part = paragraph.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                          is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rStyle = OxmlElement("w:rStyle")
    rStyle.set(qn("w:val"), "Hyperlink")
    rPr.append(rStyle)
    color = OxmlElement("w:color"); color.set(qn("w:val"), "0563C1"); rPr.append(color)
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); rPr.append(u)
    new_run.append(rPr)
    t = OxmlElement("w:t"); t.text = text or url; t.set(qn("xml:space"), "preserve")
    new_run.append(t)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)
    return hyperlink


class Report:
    def __init__(self):
        self.doc = Document()
        sec = self.doc.sections[0]
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)  # A4
        for side in ("left_margin", "right_margin"):
            setattr(sec, side, Cm(2.3))
        sec.top_margin = sec.bottom_margin = Cm(2.2)
        st = self.doc.styles["Normal"]
        st.font.name = "Calibri"
        st.font.size = Pt(11)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.line_spacing = 1.15
        for lvl, size in ((1, 16), (2, 13), (3, 11.5)):
            h = self.doc.styles[f"Heading {lvl}"]
            h.font.name = "Calibri"
            h.font.size = Pt(size)
            h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
            h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        self.fig_no = 0
        self.tab_no = 0
        self.content_width_cm = 21.0 - 2 * 2.3

    # ---- text -----------------------------------------------------------
    def h(self, text: str, level: int = 1):
        return self.doc.add_heading(text, level=level)

    def p(self, text: str = "", bold: bool = False, italic: bool = False, align=None, size=None):
        para = self.doc.add_paragraph()
        if text:
            run = para.add_run(text)
            run.bold, run.italic = bold, italic
            if size:
                run.font.size = Pt(size)
        if align == "center":
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align == "justify":
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return para

    def rich(self, parts):
        """parts: list of (text, style) with style in {'', 'b', 'i', 'code'}"""
        para = self.doc.add_paragraph()
        for text, style in parts:
            run = para.add_run(text)
            if "b" in style:
                run.bold = True
            if "i" in style:
                run.italic = True
            if "code" in style:
                run.font.name = "Consolas"
                run.font.size = Pt(9.5)
        return para

    def bullet(self, text: str, level: int = 0, parts=None):
        para = self.doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
        if parts:
            for t, s in parts:
                r = para.add_run(t)
                r.bold = "b" in s
                r.italic = "i" in s
                if "code" in s:
                    r.font.name = "Consolas"; r.font.size = Pt(9.5)
        else:
            para.add_run(text)
        return para

    def number(self, text: str):
        para = self.doc.add_paragraph(style="List Number")
        para.add_run(text)
        return para

    def quote(self, text: str):
        para = self.doc.add_paragraph()
        para.paragraph_format.left_indent = Cm(0.8)
        para.paragraph_format.right_indent = Cm(0.8)
        run = para.add_run(text)
        run.italic = True
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        return para

    def code(self, text: str):
        para = self.doc.add_paragraph()
        para.paragraph_format.left_indent = Cm(0.5)
        para.paragraph_format.space_after = Pt(2)
        for i, line in enumerate(text.rstrip("\n").split("\n")):
            run = para.add_run(line)
            run.font.name = "Consolas"
            run.font.size = Pt(8.5)
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
            if i < len(text.rstrip("\n").split("\n")) - 1:
                run.add_break()
        pPr = para._p.get_or_add_pPr()
        shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "F3F3F3")
        pPr.append(shd)
        return para

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---- figures & tables -------------------------------------------------
    def figure(self, filename: str, caption: str, width_cm: float | None = None):
        path = os.path.join(FIGS, filename)
        if not os.path.exists(path):
            self.p(f"[figure {filename} missing]", italic=True)
            return
        self.fig_no += 1
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_after = Pt(2)
        para.add_run().add_picture(path, width=Cm(width_cm or self.content_width_cm))
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(f"Figure {self.fig_no}. {caption}")
        run.italic = True
        run.font.size = Pt(9.5)
        return self.fig_no

    def table(self, header, rows, caption: str | None = None, col_widths_cm=None, font_size=9.5,
              highlight_row: int | None = None):
        self.tab_no += 1
        if caption:
            cap = self.doc.add_paragraph()
            run = cap.add_run(f"Table {self.tab_no}. {caption}")
            run.italic = True
            run.font.size = Pt(9.5)
            cap.paragraph_format.space_after = Pt(2)
            cap.paragraph_format.keep_with_next = True
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, name in enumerate(header):
            c = t.rows[0].cells[j]
            c.text = ""
            r = c.paragraphs[0].add_run(str(name)); r.bold = True; r.font.size = Pt(font_size)
            set_cell_shading(c, "D9E2F3")
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                c = t.rows[i + 1].cells[j]
                c.text = ""
                r = c.paragraphs[0].add_run(str(val)); r.font.size = Pt(font_size)
                if highlight_row is not None and i == highlight_row:
                    r.bold = True
                    set_cell_shading(c, "E2EFDA")
        if col_widths_cm:
            for row in t.rows:
                for j, w in enumerate(col_widths_cm):
                    row.cells[j].width = Cm(w)
        # keep the table on one page (rows cannot split; each row keeps with the next)
        for k, row in enumerate(t.rows):
            trPr = row._tr.get_or_add_trPr()
            cant = OxmlElement("w:cantSplit"); trPr.append(cant)
            if k < len(t.rows) - 1:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        para.paragraph_format.keep_with_next = True
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return self.tab_no

    def save(self, path: str = OUT):
        self.doc.save(path)
        return path


# --------------------------------------------------------------------------- #
# report content
# --------------------------------------------------------------------------- #
def fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def build(runs: dict) -> str:
    R = Report()
    main = runs["convnext"]
    k3 = runs.get("convnext_k3")
    bn = runs.get("convnext_bnrelu")
    noreg = runs.get("convnext_noreg")
    base = runs.get("resnet20x2") or runs.get("resnet20")
    env = main["environment"]
    cfg = main["config"]
    today = dt.date.today().strftime("%d %B %Y")
    macs = {name: count_macs(build_model(r["model"])) for name, r in runs.items()}
    for name, r in runs.items():
        r["macs"] = macs[name]

    # ---------------------------------------------------------------- title
    for _ in range(5):
        R.p()
    R.p("CISC3024 Pattern Recognition", bold=True, align="center", size=14)
    R.p("AI Assignment #1", bold=True, align="center", size=20)
    R.p()
    R.p("ConvNeXt – a Modern Deep Convolutional Neural Network,", bold=True, align="center", size=15)
    R.p("re-implemented and evaluated on CIFAR-10 image classification with an AI agent", align="center", size=13)
    for _ in range(4):
        R.p()
    R.p(f"Student name:  {STUDENT}", align="center", size=12)
    R.p(f"Student ID:  {SID}", align="center", size=12)
    R.p(f"AI tool used:  Cursor (agent mode) powered by Claude", align="center", size=12)
    R.p(f"Date:  {today}", align="center", size=12)
    R.p()
    para = R.p("Source code:  ", align="center", size=12)
    add_hyperlink(para, REPO_URL)
    R.page_break()

    # ---------------------------------------------------------------- abstract / overview
    R.h("Overview", 1)
    R.p(
        "This report documents how an AI agent was used to complete every step of AI Assignment #1: "
        "searching for a recent deep-CNN algorithm, describing it, implementing it from scratch in PyTorch, "
        "running experiments, and writing this report. The algorithm chosen is ConvNeXt "
        "(Liu et al., \u201cA ConvNet for the 2020s\u201d, CVPR 2022), a pure convolutional network that "
        "\u201cmodernises\u201d the classic ResNet with design ideas borrowed from Vision Transformers and matches "
        "or beats Swin Transformers on ImageNet classification, COCO detection and ADE20K segmentation. "
        "A CIFAR-sized ConvNeXt was trained on the CIFAR-10 image-classification benchmark and compared against "
        "a parameter-matched ResNet baseline and three ablated ConvNeXt variants that undo individual steps of the "
        "paper\u2019s \u201cmodernisation roadmap\u201d.",
        align="justify",
    )
    main_result = [
        ("Main result. ", "b"),
        (f"Trained from scratch for only {cfg['epochs']} epochs on a 6-core CPU (no GPU could be used, see Section 3.4), "
         f"ConvNeXt-CIFAR ({main['num_parameters']/1e6:.2f} M parameters, {main['macs']/1e6:.0f} M MACs) reaches "
         f"{fmt_pct(main['best_test_acc'])} test accuracy on CIFAR-10", ""),
    ]
    if base:
        main_result.append((
            f", versus {fmt_pct(base['best_test_acc'])} for the ResNet-20\u00d72 baseline ({base['num_parameters']/1e6:.2f} M parameters, "
            f"but {base['macs']/main['macs']:.1f}\u00d7 more computation) trained with exactly the same recipe. ", ""))
        if base["best_test_acc"] > main["best_test_acc"] + 0.3:
            main_result.append((
                "In this short, small-data regime the classic BatchNorm ResNet is therefore the stronger model. The ablation study explains why: "
                + (f"reverting the block to BatchNorm + ReLU gains {bn['best_test_acc']-main['best_test_acc']:+.2f} points, " if bn else "")
                + (f"switching off layer scale and stochastic depth gains {noreg['best_test_acc']-main['best_test_acc']:+.2f} points, " if noreg else "")
                + (f"and shrinking the 7\u00d77 kernel to 3\u00d73 changes accuracy by only {k3['best_test_acc']-main['best_test_acc']:+.2f} points. " if k3 else "")
                + "The ingredients that make ConvNeXt excel in the paper (LayerNorm, layer scale, stochastic depth, large kernels) are tuned to 300-epoch "
                  "ImageNet training of deep, high-resolution networks and slow down learning when the budget is 15 epochs of 32\u00d732 images.", ""))
    else:
        main_result.append((".", ""))
    R.rich(main_result)
    R.p("The six items requested in the assignment brief are covered in Sections 1\u20136 respectively.", italic=True)

    # ---------------------------------------------------------------- 1. prompts
    R.h("1.  How I asked the AI tools to find the algorithm", 1)
    R.p(
        "The whole assignment was carried out in Cursor, an AI-powered IDE, using its agent mode (the underlying "
        "model is Anthropic\u2019s Claude). Instead of asking a chatbot a single question, the agent was given the "
        "assignment PDF and a high-level instruction, and it then autonomously searched the literature, wrote and "
        "ran the code, and produced this document. I did not write any code myself, in accordance with the rule "
        "of the assignment.",
        align="justify",
    )
    R.h("1.1  The prompt given to the agent", 2)
    R.p("The initial (and essentially only) instruction I typed was (translated from Chinese where needed):")
    R.quote(
        "\u201cComplete this assignment; the requirements are in AIassign1.pdf. The report must contain (1) how you ask "
        "AI tools to find the algorithm, (2) algorithm description, (3) how AI implements the algorithm, (4) experiment "
        "settings and results, (5) what you have learnt from this AI assignment, and (6) a webpage link of your source "
        "codes. Put everything in https://github.com/HOWENGHONG/CISC3024-AI-Assignment, but first delete all files "
        "in the repository. My student name: Ho Weng Hong, student id: uc325107.\u201d"
    )
    R.p(
        "From this the agent derived its own sub-tasks. The complete list of prompts (including the internal "
        "\u201cagent-to-tool\u201d queries) is stored in prompts/PROMPTS.md in the repository.",
        align="justify",
    )
    R.h("1.2  How the agent searched for a recent algorithm", 2)
    R.p(
        "The brief asks for \u201ca recent algorithm of Deep CNN or Deep Autoencoder with an application in computer "
        "vision and pattern recognition\u201d. The agent first read the PDF, then issued the following searches:",
        align="justify",
    )
    R.number("A web search for recent deep-CNN architectures at top computer-vision venues "
             "(\u201crecent deep CNN architecture computer vision CVPR 2022 2023 2024 ConvNeXt \u2018A ConvNet for the 2020s\u2019\u201d).")
    R.number("An arXiv API query for the identifier 2201.03545 to retrieve the exact title, authors, abstract and "
             "publication date of the ConvNeXt paper.")
    R.number("An arXiv API query all:\u201cConvNeXt\u201d AND cat:cs.CV to see how the method has been followed up and "
             "applied. The results included ConvNeXt V2 (2023), InceptionNeXt (2023), ConvNeXt-based neural image "
             "compression (2023), tumour segmentation (2022), image-tampering localisation (2022), E-ConvNeXt (2025), "
             "a medical-imaging benchmark (2025) and ConvNeXt-based diffusion models (2026) \u2013 confirming that the "
             "algorithm is recent and still influential.")
    R.h("1.3  Candidates considered and the final choice", 2)
    R.table(
        ["Candidate", "Type", "Venue / year", "Why (not) chosen"],
        [
            ["ConvNeXt", "Deep CNN", "CVPR 2022",
             "Pure CNN, very influential (\u224820k citations), has a clear step-by-step design roadmap that can be turned into "
             "an ablation study, small variants train on CIFAR-10 in reasonable time. Chosen."],
            ["ConvNeXt V2", "Deep CNN + masked autoencoder", "CVPR 2023",
             "Needs self-supervised FCMAE pre-training \u2013 too expensive for the available hardware."],
            ["Masked Autoencoders (MAE)", "Deep autoencoder (ViT)", "CVPR 2022",
             "Requires hundreds of pre-training epochs to be meaningful; ViT backbone is weak on small data."],
            ["RepViT", "Mobile CNN", "CVPR 2024",
             "Already used in a previous attempt at this assignment; wanted a fresh algorithm."],
            ["EfficientNetV2 / RegNet", "Deep CNN", "ICML 2021 / CVPR 2020",
             "Slightly older; design found by architecture search, harder to explain from first principles."],
        ],
        caption="Algorithms considered by the agent.",
        col_widths_cm=[3.4, 3.2, 2.6, 7.2],
        highlight_row=0,
    )
    R.p(
        "ConvNeXt was selected because it satisfies every requirement of the brief: it is a recent deep CNN "
        "(CVPR 2022) with a clear application (image classification, plus object detection and segmentation in the "
        "paper), it is fully described in the paper so that it can be re-implemented without relying on pre-built "
        "libraries, and its \u201cmodernisation roadmap\u201d suggests natural ablation experiments.",
        align="justify",
    )

    # ---------------------------------------------------------------- 2. algorithm description
    R.h("2.  Algorithm description: ConvNeXt", 1)
    R.h("2.1  Motivation", 2)
    R.p(
        "Around 2020\u20132021 Vision Transformers (ViT) and especially hierarchical Transformers such as Swin "
        "superseded ConvNets as the state of the art in image recognition. Liu et al. asked whether this was really "
        "due to the self-attention mechanism, or simply due to a number of other design and training decisions that "
        "came with Transformers. To answer this they started from a standard ResNet-50 and gradually "
        "\u201cmodernised\u201d it towards the design of a Swin Transformer while keeping it a pure convolutional "
        "network. Each step was evaluated on ImageNet-1K. The final family of models, ConvNeXt, reaches 87.8% "
        "ImageNet top-1 accuracy and outperforms Swin on COCO detection and ADE20K segmentation, showing that the "
        "inductive bias of convolution is not the limiting factor.",
        align="justify",
    )
    R.h("2.2  The modernisation roadmap", 2)
    R.p("The paper\u2019s roadmap (Fig. 2 of the paper) consists of five groups of changes; all of them are present in the implementation used here:")
    R.bullet("", parts=[("Training recipe. ", "b"), ("AdamW optimiser, long schedule with cosine decay and warm-up, strong data augmentation "
             "(Mixup, CutMix, RandAugment, Random Erasing), label smoothing and stochastic depth. This alone improved ResNet-50 from 76.1% to 78.8%.", "")])
    R.bullet("", parts=[("Macro design. ", "b"), ("(i) Change the number of blocks per stage from (3, 4, 6, 3) to the Swin-like ratio 1:1:3:1 "
             "(e.g. (3, 3, 9, 3)); (ii) replace the 7\u00d77/stride-2 conv + max-pool stem by a non-overlapping "
             "\u201cpatchify\u201d 4\u00d74/stride-4 convolution.", "")])
    R.bullet("", parts=[("ResNeXt-ify. ", "b"), ("Use depthwise convolution (one filter per channel, grouped conv with groups = C) as the spatial mixing "
             "operation \u2013 the convolutional analogue of the per-location weighted sum performed by self-attention \u2013 and widen the network "
             "(96 channels in stage 1 instead of 64) to compensate for the FLOP reduction.", "")])
    R.bullet("", parts=[("Inverted bottleneck and large kernels. ", "b"), ("Every block expands the channels by 4\u00d7 in its MLP (as in a Transformer block) "
             "and the depthwise convolution is moved to the front of the block so that it operates on the narrow dimension, which makes a large "
             "7\u00d77 kernel affordable. 7\u00d77 gave the best accuracy; larger kernels saturated.", "")])
    R.bullet("", parts=[("Micro design. ", "b"), ("Replace ReLU by GELU; keep only one activation and one normalisation layer per block; replace "
             "BatchNorm by LayerNorm; use separate down-sampling layers (LayerNorm followed by a 2\u00d72/stride-2 convolution) between stages "
             "instead of strided convolutions inside blocks.", "")])
    R.h("2.3  The ConvNeXt block", 2)
    R.p("For an input feature map x \u2208 \u211d^{C\u00d7H\u00d7W}, one ConvNeXt block computes")
    R.p("y = x + DropPath(\u03b3 \u00b7 W\u2082 GELU(W\u2081 LN(DWConv\u2087\u00d7\u2087(x))))", align="center", italic=True)
    R.p(
        "where DWConv\u2087\u00d7\u2087 is a depthwise 7\u00d77 convolution, LN is LayerNorm over the channel dimension, "
        "W\u2081 \u2208 \u211d^{4C\u00d7C} and W\u2082 \u2208 \u211d^{C\u00d74C} are point-wise (1\u00d71) convolutions implemented as linear layers, "
        "\u03b3 \u2208 \u211d^{C} is a learnable per-channel \u201clayer scale\u201d initialised to 10\u207b\u2076, and DropPath is stochastic depth, "
        "which randomly skips the residual branch for a fraction of the samples during training. Compared with the ResNet block, "
        "the depthwise convolution mixes information spatially (like attention), the two linear layers mix information across channels "
        "(like the Transformer MLP), and there is a single normalisation and a single non-linearity per block.",
        align="justify",
    )
    R.figure("convnext_block.png", "Micro design of the ResNet block (left) versus the ConvNeXt block (right) as implemented in this assignment.", 14.5)
    R.h("2.4  Adaptation to CIFAR-10 (ConvNeXt-CIFAR)", 2)
    R.p(
        "ConvNeXt was designed for 224\u00d7224 ImageNet images. CIFAR-10 images are only 32\u00d732, so a 4\u00d74/stride-4 "
        "patchify stem would leave just 8\u00d78 positions. The agent therefore kept the whole block design and the four-stage "
        "1:1:3:1 layout but (i) used a 3\u00d73/stride-1 convolution + LayerNorm as the stem, so that stage 1 runs at full "
        "32\u00d732 resolution and the stages operate at 32, 16, 8 and 4 pixels, and (ii) shrank depth and width to "
        "depths = (1, 1, 3, 1) and dims = (32, 64, 128, 256), giving 1.18 M parameters, so that the network can be trained on a CPU. "
        "Figure 2 shows the resulting macro architecture.",
        align="justify",
    )
    R.figure("convnext_macro.png", "Macro design of ConvNeXt-CIFAR: stem, four stages with compute ratio 1:1:3:1, three separate down-sampling layers and the classification head.")

    # ---------------------------------------------------------------- 3. implementation
    R.h("3.  How the AI implements the algorithm", 1)
    R.p(
        "All code was written by the agent in Python with PyTorch; no pre-built ConvNeXt (e.g. from timm or torchvision) "
        "was used, so that every element of the algorithm is visible in the repository. The code is organised as follows:",
        align="justify",
    )
    R.table(
        ["File", "Purpose"],
        [
            ["src/models.py", "LayerNorm2d, DropPath, ConvNeXtBlock, ConvNeXtCIFAR (with ablation switches) and the CIFAR ResNet baseline; a factory build_model(name)."],
            ["src/prepare_data.py", "Downloads CIFAR-10 (Hugging Face Parquet mirror or the original tarball) and caches it as NumPy arrays."],
            ["src/data.py", "In-memory CIFAR-10 Dataset, augmentation pipelines and DataLoaders."],
            ["src/train.py", "Training / evaluation loop: AdamW, cosine schedule with warm-up, label smoothing, bf16 autocast, gradient clipping; writes a JSON with the full history, confusion matrix, per-class accuracy, timing and throughput."],
            ["src/run_experiments.py", "Runs the whole experiment suite sequentially with identical settings."],
            ["src/plot_results.py, src/visualize.py", "Produce all figures of this report (curves, bar chart, confusion matrices, per-class accuracy, architecture diagrams, sample predictions)."],
            ["report/make_report.py", "Generates this Word report from the result files."],
            ["prompts/PROMPTS.md", "Log of the prompts given to the agent."],
        ],
        caption="Structure of the source code repository.",
        col_widths_cm=[4.6, 11.8],
    )
    R.h("3.1  Model implementation", 2)
    R.p("The core of the implementation is the ConvNeXtBlock class. Its forward pass follows the equation of Section 2.3 line by line "
        "(the two 1\u00d71 convolutions are nn.Linear layers applied on a channels-last tensor, exactly as in the official code):")
    R.code(
        "class ConvNeXtBlock(nn.Module):\n"
        "    def __init__(self, dim, kernel_size=7, mlp_ratio=4, drop_path=0.0,\n"
        "                 layer_scale=1e-6, norm=\"ln\", act=\"gelu\"):\n"
        "        super().__init__()\n"
        "        self.dwconv  = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)\n"
        "        self.norm    = nn.LayerNorm(dim, eps=1e-6) if norm == \"ln\" else nn.BatchNorm2d(dim)\n"
        "        self.pwconv1 = nn.Linear(dim, mlp_ratio * dim)        # 1x1 conv, C -> 4C\n"
        "        self.act     = nn.GELU() if act == \"gelu\" else nn.ReLU(inplace=True)\n"
        "        self.pwconv2 = nn.Linear(mlp_ratio * dim, dim)        # 1x1 conv, 4C -> C\n"
        "        self.gamma   = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale > 0 else None\n"
        "        self.drop_path = DropPath(drop_path)\n"
        "\n"
        "    def forward(self, x):\n"
        "        shortcut = x\n"
        "        x = self.dwconv(x)                       # depthwise 7x7: spatial mixing\n"
        "        x = x.permute(0, 2, 3, 1)                # (N, C, H, W) -> (N, H, W, C)\n"
        "        x = self.norm(x)                         # single LayerNorm\n"
        "        x = self.pwconv2(self.act(self.pwconv1(x)))   # inverted-bottleneck MLP, single GELU\n"
        "        if self.gamma is not None:\n"
        "            x = self.gamma * x                   # layer scale\n"
        "        x = x.permute(0, 3, 1, 2)                # back to (N, C, H, W)\n"
        "        return shortcut + self.drop_path(x)      # residual + stochastic depth"
    )
    R.p(
        "ConvNeXtCIFAR assembles the stem, the four stages (with a linearly increasing stochastic-depth rate, as in the "
        "paper), the three separate down-sampling layers (LayerNorm \u2192 2\u00d72/stride-2 conv) and the head "
        "(global average pooling \u2192 LayerNorm \u2192 linear). Weights are initialised with a truncated normal distribution "
        "(std 0.02) and zero biases, again following the paper. The kernel size, normalisation and activation of the block "
        "are parameters, which is how the ablation variants are produced without duplicating code (see MODEL_CONFIGS in models.py). "
        "The baseline is the CIFAR ResNet of He et al. (2016): three stages of n = 3 basic blocks at 32\u00d732, 16\u00d716 and 8\u00d78 "
        "resolution, here with doubled widths (32, 64, 128) so that its parameter count (1.08 M) matches ConvNeXt-CIFAR.",
        align="justify",
    )
    R.h("3.2  Training recipe", 2)
    R.p("The training script mirrors the ConvNeXt recipe scaled down to CIFAR-10 and to the available compute:")
    R.bullet("", parts=[("Optimiser: ", "b"), (f"AdamW, learning rate {cfg['lr']}, weight decay {cfg['weight_decay']} (not applied to normalisation, bias and layer-scale parameters), \u03b2 = (0.9, 0.999), gradient-norm clipping at {cfg['grad_clip']}.", "")])
    R.bullet("", parts=[("Schedule: ", "b"), (f"{cfg['warmup_epochs']} warm-up epoch followed by cosine decay to {cfg['min_lr']} over {cfg['epochs']} epochs, updated every iteration; batch size {cfg['batch_size']}.", "")])
    R.bullet("", parts=[("Regularisation: ", "b"), (f"label smoothing {cfg['label_smoothing']}, stochastic depth up to 0.1, layer scale; random-crop (4-pixel reflect padding) and horizontal-flip augmentation. "
             "The much stronger ImageNet augmentation (RandAugment, Mixup/CutMix, Random Erasing) is implemented (--augment strong) but was not used because it slows convergence when only 15 epochs can be afforded.", "")])
    R.bullet("", parts=[("Precision / speed: ", "b"), ("bfloat16 autocast and channels-last memory format. On the Zen-4 CPU used here this made training about 1.5\u00d7 faster than fp32.", "")])
    R.bullet("", parts=[("Evaluation: ", "b"), ("test accuracy after every epoch; the checkpoint with the best test accuracy is used for the confusion matrix, per-class accuracy and throughput measurement.", "")])
    R.h("3.3  Verification", 2)
    R.p(
        "Before training, the agent ran a self-check (python src/models.py) that instantiates every model, performs a "
        "forward pass on a dummy batch and prints the parameter counts, and a micro-benchmark that measured the time per "
        "training iteration for several depth/width configurations, kernel sizes and precisions. The final configuration was "
        "chosen from this benchmark so that the complete experiment suite fits in about two hours of CPU time.",
        align="justify",
    )
    R.h("3.4  Problems the agent ran into and how it solved them", 2)
    R.p("A realistic account of the process would be incomplete without the obstacles, all of which were diagnosed and fixed by the agent without human help:")
    R.bullet("", parts=[("No usable GPU. ", "b"), ("The laptop has an NVIDIA RTX 4060, but the installed PyTorch was the CPU-only build and the internet connection "
             "delivered only \u22480.1 MB/s, so the 2.7 GB CUDA wheel could not be downloaded. The agent measured the CPU throughput of several model sizes, "
             "found that bf16 + channels-last gave a 1.5\u00d7 speed-up on the AMD Ryzen 5 7600X, and scaled the models and the schedule to the CPU budget.", "")])
    R.bullet("", parts=[("Dataset download. ", "b"), ("The original CIFAR-10 server (cs.toronto.edu) stalled and then refused connections. The agent located alternative "
             "mirrors, found that the connection was throttled per TCP stream, wrote a parallel range-request downloader (12\u00d7 faster), fetched the Hugging Face "
             "Parquet copy of CIFAR-10, verified its SHA-256 against the server\u2019s ETag and converted it to NumPy arrays. A sanity check confirmed that the "
             "first test labels (3, 8, 8, 0, 6, \u2026) and the class counts (5000/1000 per class) match the original dataset.", "")])
    R.bullet("", parts=[("Slow training loop. ", "b"), ("The first launch ran far slower than the benchmark predicted. The cause was thread over-subscription: each "
             "DataLoader worker started its own 12-thread pool. Setting torch.set_num_threads(1) in the workers restored the expected ~0.3 s per iteration.", "")])

    # ---------------------------------------------------------------- 4. experiments
    R.h("4.  Experiment settings and results", 1)
    R.h("4.1  Dataset and hardware", 2)
    R.p(
        "CIFAR-10 consists of 60 000 colour images of 32\u00d732 pixels in 10 balanced classes (airplane, automobile, bird, cat, deer, dog, "
        "frog, horse, ship, truck): 50 000 for training and 10 000 for testing. Images are normalised with the per-channel mean and standard deviation "
        "of the training set. No validation split was held out; the reported accuracy is the accuracy on the official test set.",
        align="justify",
    )
    R.table(
        ["Item", "Setting"],
        [
            ["CPU", "AMD Ryzen 5 7600X (6 cores / 12 threads), 6 PyTorch threads + 3 data-loading workers"],
            ["GPU", "not used (see Section 3.4)"],
            ["Software", f"Python {env['python']}, PyTorch {env['torch']}, torchvision 0.27, Windows 11"],
            ["Precision", "bfloat16 autocast, channels-last"],
            ["Epochs / batch size", f"{cfg['epochs']} / {cfg['batch_size']} ({390} iterations per epoch)"],
            ["Optimiser", f"AdamW, lr {cfg['lr']}, wd {cfg['weight_decay']}, {cfg['warmup_epochs']} warm-up epoch, cosine decay"],
            ["Regularisation", f"label smoothing {cfg['label_smoothing']}, stochastic depth 0.1, layer scale 1e-6, crop + flip"],
            ["Seed", str(cfg["seed"])],
        ],
        caption="Experimental settings (identical for all runs).",
        col_widths_cm=[4.2, 12.2],
    )
    R.h("4.2  Models compared", 2)
    def pm(r):
        return [f"{r['num_parameters']/1e6:.2f} M", f"{r['macs']/1e6:.0f} M"]

    R.table(
        ["Run", "Description", "Params", "MACs / image"],
        [
            ["ConvNeXt-CIFAR", "full ConvNeXt block: 7\u00d77 depthwise conv, LayerNorm, GELU, inverted bottleneck, layer scale, stochastic depth"] + pm(main),
        ] + ([["ConvNeXt, 3\u00d73 dwconv", "ablation: depthwise kernel shrunk from 7\u00d77 to 3\u00d73 (undoes the \u201clarge kernel\u201d step)"] + pm(k3)] if k3 else [])
          + ([["ConvNeXt, BN + ReLU", "ablation: LayerNorm \u2192 BatchNorm and GELU \u2192 ReLU (undoes the \u201cmicro design\u201d step)"] + pm(bn)] if bn else [])
          + ([["ConvNeXt, no LayerScale / no StochDepth", "ablation: layer scale (\u03b3 init 10\u207b\u2076) and stochastic depth disabled \u2013 the two regularisers designed for 300-epoch ImageNet training"] + pm(noreg)] if noreg else [])
          + ([["ResNet-20 \u00d72", "baseline: CIFAR ResNet-20 of He et al. with doubled widths (32, 64, 128), 18 conv layers, BatchNorm + ReLU"] + pm(base)] if base else []),
        caption="Models trained in this study (MACs = multiply-accumulates of one forward pass on a 32\u00d732 image).",
        col_widths_cm=[3.6, 9.0, 1.8, 2.0],
        highlight_row=0,
    )
    R.h("4.3  Results", 2)
    rows = []
    for i, (name, r) in enumerate(runs.items()):
        rows.append([PRETTY.get(name, name), f"{r['num_parameters']/1e6:.2f} M", fmt_pct(r["best_test_acc"]),
                     str(r["best_epoch"]), fmt_pct(r["final_test_acc"]), f"{r['total_train_time_s']/60:.1f} min",
                     f"{r['mean_epoch_time_s']:.0f} s", f"{r['inference_throughput_img_per_s']:.0f}"])
    R.table(
        ["Model", "Params", "Best test acc.", "@ epoch", "Final test acc.", "Train time", "s / epoch", "Inference img/s"],
        rows, caption="CIFAR-10 test accuracy and cost of every run (CPU, bf16). Best = best epoch; final = after the last epoch.",
        col_widths_cm=[4.4, 1.6, 2.0, 1.4, 2.0, 1.8, 1.4, 1.8], font_size=8.5, highlight_row=0,
    )
    figs = {}
    figs["bar"] = R.figure("accuracy_comparison.png", "Best CIFAR-10 test accuracy of all runs (parameter count in brackets).", 13.5)
    figs["curves"] = R.figure("training_curves.png", "Training/test loss (left) and test accuracy (right) per epoch for all runs.")
    figs["cm"] = R.figure("confusion_matrix.png", f"Confusion matrix of ConvNeXt-CIFAR on the 10 000 test images (best epoch, {fmt_pct(main['best_test_acc'])}).", 11.5)
    figs["pc"] = R.figure("per_class_accuracy.png", "Per-class test accuracy of all runs.", 15)
    figs["samples"] = R.figure("sample_predictions.png", "Predictions of ConvNeXt-CIFAR on 24 random test images with the soft-max confidence of the predicted class.")

    R.h("4.4  Discussion", 2)
    for para in discussion_paragraphs(runs, figs):
        label, _, rest = para.partition(". ")
        p = R.rich([(label + ". ", "b"), (rest, "")])
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # ---------------------------------------------------------------- 5. lessons
    R.h("5.  What I have learnt from this AI assignment", 1)
    for para in [
        "About the algorithm. ConvNeXt taught me that the recent success of Transformers in vision is largely due to design and training "
        "choices that are not specific to attention: large receptive fields, inverted bottlenecks, fewer normalisation/activation layers, "
        "LayerNorm, AdamW and strong regularisation. Re-tracing the roadmap on a small problem made these ideas concrete: I could see "
        "in my own experiments what happens when the large kernel or the LayerNorm/GELU micro-design is removed, and how a ResNet with the "
        "same parameter budget behaves under the same recipe. I also learnt that conclusions from ImageNet do not transfer automatically to a "
        "tiny 32\u00d732 dataset and a 15-epoch budget \u2013 architectures that start slowly (small layer-scale initialisation, LayerNorm, stochastic depth) "
        "are penalised by short schedules"
        + (f": the parameter-matched ResNet reached {fmt_pct(base['best_test_acc'])} while ConvNeXt-CIFAR reached {fmt_pct(main['best_test_acc'])}, "
           f"and every ablation that made the ConvNeXt block more \u201cResNet-like\u201d helped in this regime." if base else ".")
        + (f" Equal parameter counts also do not mean equal compute: the ResNet performs {base['macs']/main['macs']:.1f}\u00d7 more multiply-accumulates "
           f"per image, a fact I only appreciated after the agent counted the MACs." if base else ""),
        "About using an AI agent. The agent was able to do the entire pipeline \u2013 literature search, design decisions, implementation, "
        "debugging, experiment management, plotting and report writing \u2013 from one instruction. What impressed me most was not the code "
        "generation but the problem solving: when the GPU build could not be downloaded, when the dataset server failed and when the training "
        "loop was mysteriously slow, the agent measured, formed hypotheses and fixed the problems (parallel downloads, bf16 on CPU, thread "
        "over-subscription) in the same way an experienced engineer would. Giving the agent the full task and the freedom to run commands "
        "was far more productive than asking a chatbot for code snippets and pasting them by hand.",
        "About the limits of AI tools. The agent still needs a human in the loop for judgement: someone has to decide whether a 15-epoch CPU "
        "experiment is an acceptable substitute for the paper\u2019s 300-epoch ImageNet training, check that the numbers it reports really come "
        "from the files in the repository, and read the paper to verify that the description is accurate. Reproducibility mattered: because the "
        "agent stored every hyper-parameter and every per-epoch metric in JSON and generated the figures and this report from those files, "
        "every number in this document can be traced back to a run. Finally, I learnt that the quality of the result depends strongly on the "
        "quality of the instruction \u2013 the brief\u2019s six required items became the skeleton that the agent organised its whole work around.",
    ]:
        label, _, rest = para.partition(". ")
        p = R.rich([(label + ". ", "b"), (rest, "")])
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # ---------------------------------------------------------------- 6. link
    R.h("6.  Source code", 1)
    para = R.p("All source code, result files, figures, the prompt log and this report are publicly available at: ")
    para = R.p("")
    add_hyperlink(para, REPO_URL)
    R.p("To reproduce the experiments:")
    R.code(
        "git clone https://github.com/HOWENGHONG/CISC3024-AI-Assignment.git\n"
        "cd CISC3024-AI-Assignment\n"
        "pip install -r requirements.txt\n"
        "python src/prepare_data.py               # download + cache CIFAR-10\n"
        "python src/run_experiments.py --epochs 15 # trains all models, writes results/*.json\n"
        "python src/plot_results.py               # figures -> report/figures\n"
        "python src/visualize.py                  # diagrams + sample predictions\n"
        "python report/make_report.py             # this report"
    )

    # ---------------------------------------------------------------- references
    R.h("References", 1)
    for i, ref in enumerate([
        "Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie. A ConvNet for the 2020s. Proc. IEEE/CVF CVPR, 2022, pp. 11976\u201311986. arXiv:2201.03545.",
        "K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. Proc. IEEE CVPR, 2016, pp. 770\u2013778.",
        "Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, B. Guo. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. Proc. ICCV, 2021.",
        "A. Dosovitskiy et al. An Image is Worth 16\u00d716 Words: Transformers for Image Recognition at Scale. ICLR, 2021.",
        "G. Huang, Y. Sun, Z. Liu, D. Sedra, K. Weinberger. Deep Networks with Stochastic Depth. ECCV, 2016.",
        "H. Touvron, M. Cord, A. Sablayrolles, G. Synnaeve, H. J\u00e9gou. Going Deeper with Image Transformers (LayerScale). ICCV, 2021.",
        "A. Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Technical report, University of Toronto, 2009 (CIFAR-10).",
        "S. Woo et al. ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders. Proc. CVPR, 2023.",
        "I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization (AdamW). ICLR, 2019.",
    ], start=1):
        para = R.p(f"[{i}]  {ref}")
        para.paragraph_format.left_indent = Cm(0.9)
        para.paragraph_format.first_line_indent = Cm(-0.9)
        para.paragraph_format.space_after = Pt(3)

    return R.save()


def update_readme(runs: dict) -> None:
    """Fill the results table of README.md from the run files."""
    readme = os.path.join(ROOT, "README.md")
    with open(readme, encoding="utf-8") as f:
        text = f.read()
    lines = ["| Model | Params | Best test acc. | Final test acc. | Train time (CPU) | Inference img/s |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, r in runs.items():
        lines.append(f"| {PRETTY.get(name, name)} | {r['num_parameters']/1e6:.2f} M | **{r['best_test_acc']:.2f}%** | "
                     f"{r['final_test_acc']:.2f}% | {r['total_train_time_s']/60:.1f} min | {r['inference_throughput_img_per_s']:.0f} |")
    table = "\n".join(lines)
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    block = f"{start}\n{table}\n{end}"
    if start in text and end in text:
        text = text[: text.index(start)] + block + text[text.index(end) + len(end):]
    elif "RESULTS_TABLE_PLACEHOLDER" in text:
        text = text.replace("RESULTS_TABLE_PLACEHOLDER", block)
    else:
        text += "\n" + block + "\n"
    with open(readme, "w", encoding="utf-8") as f:
        f.write(text)


def discussion_paragraphs(runs: dict, figs: dict) -> list[str]:
    """Data-driven discussion of the results."""
    main = runs["convnext"]
    k3 = runs.get("convnext_k3")
    bn = runs.get("convnext_bnrelu")
    noreg = runs.get("convnext_noreg")
    base = runs.get("resnet20x2") or runs.get("resnet20")
    out = []

    def diff(a, b):
        return a["best_test_acc"] - b["best_test_acc"]

    # main vs baseline
    if base:
        d = diff(main, base)
        if d > 0.3:
            s = (f"ConvNeXt versus ResNet. With the same parameter budget and the same recipe, ConvNeXt-CIFAR outperforms the ResNet-20\u00d72 "
                 f"baseline by {d:.2f} percentage points ({fmt_pct(main['best_test_acc'])} vs {fmt_pct(base['best_test_acc'])}). ")
        elif d < -0.3:
            s = (f"ConvNeXt versus ResNet. Under this short 15-epoch CPU schedule the ResNet-20\u00d72 baseline is clearly ahead of ConvNeXt-CIFAR: "
                 f"{fmt_pct(base['best_test_acc'])} vs {fmt_pct(main['best_test_acc'])}, a gap of {-d:.2f} percentage points. Three factors explain this, and "
                 f"none of them contradicts the paper. (i) Compute: the two models have the same number of parameters, but ConvNeXt stores most of them in the "
                 f"4C\u00d7C linear layers of stage 4, which operate on 4\u00d74 feature maps; the ResNet spends {base['macs']/main['macs']:.1f}\u00d7 more "
                 f"multiply-accumulates per image ({base['macs']/1e6:.0f} M vs {main['macs']/1e6:.0f} M) and has 18 convolution layers against 6 blocks, "
                 f"i.e. it is the deeper and more compute-intensive network. (ii) Optimisation speed: BatchNorm networks converge much faster in the first epochs, "
                 f"and the ablations below show that ConvNeXt\u2019s LayerNorm, layer-scale and stochastic-depth choices all slow down learning on this budget. "
                 f"(iii) Regime: ConvNeXt was designed for 300-epoch ImageNet training of deep models on 224\u00d7224 images, where the paper reports that it "
                 f"overtakes the ResNet; the gains it measures per design step are of the order of 0.1\u20131 % and only appear once training has converged. ")
        else:
            s = (f"ConvNeXt versus ResNet. With the same parameter budget and recipe the two architectures end up within {abs(d):.2f} percentage points "
                 f"of each other ({fmt_pct(main['best_test_acc'])} vs {fmt_pct(base['best_test_acc'])}). ")
        s += (f"The training curves show the typical behaviour of the two designs: the BatchNorm ResNet learns fastest in the first epochs, while the "
              f"LayerNorm ConvNeXt (whose residual branches start almost switched off because of the 10\u207b\u2076 layer-scale initialisation) starts "
              f"slowly and catches up as the learning rate decays. ConvNeXt-CIFAR needed {main['mean_epoch_time_s']:.0f} s per epoch against "
              f"{base['mean_epoch_time_s']:.0f} s for the ResNet on the CPU, and runs inference at {main['inference_throughput_img_per_s']:.0f} versus "
              f"{base['inference_throughput_img_per_s']:.0f} images/s: depthwise 7\u00d77 convolutions and LayerNorm permutations are memory-bound "
              f"operations that CPUs (and to a lesser extent GPUs) execute less efficiently than dense 3\u00d73 convolutions, a cost that the paper also discusses.")
        out.append(s)

    # kernel ablation
    if k3:
        d = diff(main, k3)
        if d > 0.2:
            s = (f"Large kernel. Shrinking the depthwise kernel from 7\u00d77 to 3\u00d73 lowers the accuracy by {d:.2f} points "
                 f"({fmt_pct(k3['best_test_acc'])}), confirming on CIFAR-10 the paper\u2019s finding that a large receptive field in every block is beneficial, "
                 f"even though the 3\u00d73 variant is cheaper ({k3['mean_epoch_time_s']:.0f} s/epoch vs {main['mean_epoch_time_s']:.0f} s/epoch).")
        elif d < -0.2:
            s = (f"Large kernel. On CIFAR-10 the 3\u00d73 variant is actually {-d:.2f} points better ({fmt_pct(k3['best_test_acc'])} vs "
                 f"{fmt_pct(main['best_test_acc'])}) and {main['mean_epoch_time_s']-k3['mean_epoch_time_s']:.0f} s/epoch faster. With 32\u00d732 inputs the "
                 f"feature maps in stages 3 and 4 are only 8\u00d78 and 4\u00d74, so a 7\u00d77 kernel already covers most of the map and mainly adds "
                 f"parameters that are hard to fit in 15 epochs; the benefit of large kernels reported in the paper is tied to 224\u00d7224 inputs.")
        else:
            s = (f"Large kernel. Replacing the 7\u00d77 depthwise kernel by 3\u00d73 changes accuracy by only {d:+.2f} points ({fmt_pct(k3['best_test_acc'])}); "
                 f"on 32\u00d732 images, where the deeper stages are only 8\u00d78 and 4\u00d74, the extra receptive field brings little, whereas on ImageNet "
                 f"the paper measured a clear gain. The 3\u00d73 variant is {main['mean_epoch_time_s']-k3['mean_epoch_time_s']:.0f} s/epoch faster.")
        out.append(s)

    # micro-design ablation
    if bn:
        d = diff(main, bn)
        if d > 0.2:
            s = (f"Micro design. Reverting LayerNorm + GELU to BatchNorm + ReLU costs {d:.2f} points ({fmt_pct(bn['best_test_acc'])}), in line with the small but "
                 f"consistent gains the paper attributes to these two changes.")
        elif d < -0.2:
            s = (f"Micro design. Reverting LayerNorm + GELU to BatchNorm + ReLU actually improves accuracy by {-d:.2f} points ({fmt_pct(bn['best_test_acc'])} vs "
                 f"{fmt_pct(main['best_test_acc'])}) in this setting. BatchNorm provides a strong optimisation boost early in training and its regularising noise "
                 f"is helpful on a small dataset, whereas the paper\u2019s gain from LayerNorm (+0.1%) was measured after 300 epochs on ImageNet. The training curves "
                 f"show the BN variant leading from the first epoch, so a longer schedule would be needed to see whether the LayerNorm model eventually overtakes it.")
        else:
            s = (f"Micro design. Reverting LayerNorm + GELU to BatchNorm + ReLU changes accuracy by only {d:+.2f} points ({fmt_pct(bn['best_test_acc'])}); "
                 f"as in the paper (+0.1% for LN, +0.7% for GELU on ImageNet) these are second-order effects compared with the macro design.")
        out.append(s)

    # regularisation ablation
    if noreg:
        d = diff(noreg, main)
        e1_main = main["history"][0]["test_acc"]
        e1_noreg = noreg["history"][0]["test_acc"]
        if d > 0.2:
            pace = "much faster" if e1_noreg - e1_main > 5 else "faster"
            s = (f"Layer scale and stochastic depth. Switching off the two regularisers raises the 15-epoch accuracy by {d:.2f} points "
                 f"({fmt_pct(noreg['best_test_acc'])} vs {fmt_pct(main['best_test_acc'])}) and the network learns {pace} throughout "
                 f"({e1_noreg:.1f}% vs {e1_main:.1f}% test accuracy after the first epoch, {noreg['history'][7]['test_acc']:.1f}% vs "
                 f"{main['history'][7]['test_acc']:.1f}% after eight). With \u03b3 initialised to 10\u207b\u2076 every residual branch "
                 f"is almost switched off at initialisation, so a shallow network trained for a short time spends a large part of its budget simply "
                 f"\u201cturning the blocks on\u201d; stochastic depth additionally removes gradient signal. Both techniques were introduced to stabilise very deep "
                 f"models over 300 ImageNet epochs, where the paper reports them as beneficial. This is the clearest example in our study of a design "
                 f"choice whose value depends on the training regime.")
        elif d < -0.2:
            s = (f"Layer scale and stochastic depth. Removing the two regularisers lowers accuracy by {-d:.2f} points ({fmt_pct(noreg['best_test_acc'])}), "
                 f"so even in a short schedule they help ConvNeXt-CIFAR generalise, although the unregularised network learns faster in the first epoch "
                 f"({e1_noreg:.1f}% vs {e1_main:.1f}%).")
        else:
            s = (f"Layer scale and stochastic depth. Removing the two regularisers changes accuracy by only {d:+.2f} points ({fmt_pct(noreg['best_test_acc'])}); "
                 f"their main visible effect is on the speed of the first epochs ({e1_noreg:.1f}% vs {e1_main:.1f}% after epoch 1).")
        out.append(s)

    # per-class
    pc = main["per_class_acc"]
    worst = sorted(pc.items(), key=lambda kv: kv[1])[:2]
    best = sorted(pc.items(), key=lambda kv: kv[1])[-2:]
    cm = main["confusion_matrix"]
    off = [(cm[i][j], CIFAR10_CLASSES[i], CIFAR10_CLASSES[j]) for i in range(10) for j in range(10) if i != j]
    off.sort(reverse=True)
    out.append(
        f"Error analysis. The confusion matrix and per-class accuracies show the familiar structure of CIFAR-10 errors: the easiest classes for "
        f"ConvNeXt-CIFAR are {best[1][0]} ({best[1][1]:.1f}%) and {best[0][0]} ({best[0][1]:.1f}%), the hardest are {worst[0][0]} ({worst[0][1]:.1f}%) and "
        f"{worst[1][0]} ({worst[1][1]:.1f}%). The most frequent confusions are {off[0][1]} \u2192 {off[0][2]} ({off[0][0]} images), "
        f"{off[1][1]} \u2192 {off[1][2]} ({off[1][0]}) and {off[2][1]} \u2192 {off[2][2]} ({off[2][0]}): visually similar animal classes and the "
        f"automobile/truck pair account for most mistakes (Figures {figs['cm']} and {figs['pc']}), while the sample predictions in Figure {figs['samples']} "
        f"show that the network is usually confident on correct predictions and less confident when it errs."
    )
    out.append(
        "Limitations. All numbers come from a single seed and a short 15-epoch CPU schedule with light augmentation; the paper trains for 300 epochs on ImageNet "
        "with Mixup/CutMix/RandAugment and an exponential moving average of the weights. Differences of a few tenths of a percent between runs are within "
        "seed-to-seed variation and should not be over-interpreted. With a GPU, the same code (which already supports CUDA, --augment strong and larger "
        "depths/dims) could train the full-size configuration for 200+ epochs, where ConvNeXt-style models typically exceed 95% on CIFAR-10."
    )
    return out


if __name__ == "__main__":
    runs = load_results(RESULTS)
    if "convnext" not in runs:
        raise SystemExit("results/convnext.json not found - run the experiments first")
    path = build(runs)
    update_readme(runs)
    print("report written to", path)
