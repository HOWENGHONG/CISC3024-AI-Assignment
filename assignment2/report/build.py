import pathlib
import subprocess

here = pathlib.Path(__file__).parent
root = here.parent
md = (here / "report.md").read_text(encoding="utf-8")
md = md.replace("<<<BAPE>>>", (root / "bape.py").read_text(encoding="utf-8").rstrip())
md = md.replace("<<<TRAIN>>>", (root / "train.py").read_text(encoding="utf-8").rstrip())
built = here / "_report_built.md"
built.write_text(md, encoding="utf-8")

common = [
    "pandoc", str(built),
    "--number-sections", "--toc", "--toc-depth=2",
    "--resource-path", str(root),
]
pdf = subprocess.run(
    common + ["--listings", "--pdf-engine=pdflatex", "-o", str(here / "AIAssignment2_Report.pdf")],
    capture_output=True, text=True, cwd=here,
)
print("PDF rc", pdf.returncode)
print(pdf.stderr[-4000:])
docx = subprocess.run(
    common + ["-o", str(here / "AIAssignment2_Report.docx")],
    capture_output=True, text=True, cwd=here,
)
print("DOCX rc", docx.returncode)
print(docx.stderr[-2000:])
