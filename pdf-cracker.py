"""
Author: Davood Yahay
Website: https://davoodya.ir
GitHub: https://github.com/davoodya/Yakuza_Malwares_Arsenal/pdf-cracker
Start Date: 27 November(11) 2025 | 6 Azar(9) 1404
End Date:

- Brief Description: OCR + Searchable PDF + DOCX generator
- Converts multi-page image-based PDF to:
  1) Searchable PDF (output_searchable.pdf)
  2) DOCX with page images + extracted text (output.docx)
  3) Optional plain text (output.txt)

- Functionality: PDF-Cracker Using for:
1. Scan a Locked and Not Editable PDF Document
2. Convert each page to an Image
3. Run OCR on All Images Pages
4. Split Text Image and Describe Image
5. Store New Editable PDF Document + Word(DOCX) + TEXT Version of Original PDF

- Requirements:
  - Tesseract OCR installed (https://github.com/tesseract-ocr/tesseract)
  - pip install pdf2image pytesseract python-docx pillow opencv-python


- TODO:

"""

import os
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output
from PIL import Image
import cv2
from docx import Document
from docx.shared import Inches

# ---------- CONFIG ----------
PDF_INPUT = "H:\test\DataRemoval.pdf"
OUT_PREFIX = "output"
DPI = 300
LANG = "eng"      # change to "fas" or "eng+fas" if you have Persian trained data
USE_DESKEW = True
POPPLER_PATH = r"H:\Repo\LittleApps\Materials\poppler\Library\bin"
TESSERACT_CMD = r"H:\Repo\LittleApps\Materials\Tesseract-OCR\tesseract\tesseract.exe"
# TESSERACT_CMD = None if in PATH or  # e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ----------------------------

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

work_dir = Path("ocr_temp")
work_dir.mkdir(exist_ok=True)

# helper: simple deskew and denoise using OpenCV
def preprocess_image(pil_image):
    # convert to OpenCV format
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # threshold (adaptive)
    th = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 41, 11)
    if USE_DESKEW:
        coords = cv2.findNonZero(255 - th)
        if coords is not None:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = th.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(th, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            th = rotated
    return Image.fromarray(th)

# import numpy after function to keep top imports minimal
import numpy as np

# Convert PDF to images
print("Converting PDF to images...")
pages = convert_from_path(PDF_INPUT, dpi=DPI, poppler_path=POPPLER_PATH)

doc = Document()
all_texts = []

searchable_pdf_bytes = []

# We'll build a searchable PDF by letting tesseract output PDF for each page,
# then combine bytes of pages into final PDF.
# pytesseract.image_to_pdf_or_hocr returns PDF bytes if extension='pdf'
pdf_pages_bytes = []

for i, page in enumerate(pages, start=1):
    print(f"Processing page {i}/{len(pages)}...")
    # optional preprocessing
    proc_img = preprocess_image(page)

    # Save preprocessed image (optional)
    img_path = work_dir / f"page_{i:03d}.png"
    proc_img.save(img_path)

    # OCR text extraction
    custom_config = r'--oem 3 --psm 3'  # adjust psm for whole page; psm 1..13 options
    text = pytesseract.image_to_string(proc_img, lang=LANG, config=custom_config)
    text = text.strip()
    all_texts.append(text)

    # Create page-level searchable PDF bytes
    pdf_bytes = pytesseract.image_to_pdf_or_hocr(proc_img, lang=LANG, extension='pdf')
    pdf_pages_bytes.append(pdf_bytes)

    # Add to DOCX: image + text
    doc.add_paragraph(f"--- Page {i} ---")
    # insert image (scale to page width; adjust Inches as needed)
    doc.add_picture(str(img_path), width=Inches(6))
    doc.add_paragraph("")  # spacer
    doc.add_paragraph(text)

# Combine per-page PDF bytes into a single searchable PDF file
# Simple method: write bytes sequentially — but PDF merging requires a proper merger.
# Use pypdf to merge page PDFs reliably.
try:
    from pypdf import PdfMerger
except Exception:
    from PyPDF2 import PdfFileMerger as PdfMerger

merger = PdfMerger()
for idx, pb in enumerate(pdf_pages_bytes):
    tmp_pdf = work_dir / f"_tmp_page_{idx}.pdf"
    tmp_pdf.write_bytes(pb)
    merger.append(str(tmp_pdf))

out_pdf_path = f"{OUT_PREFIX}_searchable.pdf"
merger.write(out_pdf_path)
merger.close()

# Save DOCX
out_docx_path = f"{OUT_PREFIX}.docx"
doc.save(out_docx_path)

# Save plain text as well
out_txt_path = f"{OUT_PREFIX}.txt"
with open(out_txt_path, "w", encoding="utf-8") as f:
    f.write("\n\n==== PAGE BREAK ==== \n\n".join(all_texts))

print("Done!")
print(f"Searchable PDF: {out_pdf_path}")
print(f"DOCX (editable): {out_docx_path}")
print(f"Plain text: {out_txt_path}")
print(f"Temp images in: {work_dir}")

