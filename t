import fitz  # PyMuPDF
import hashlib
import os
from PIL import Image
import io

# ---------- CONFIG ----------
PDF_PATH = "input.pdf"
OUTPUT_DIR = "output_images"
MIN_IMAGE_SIZE = 100  # px
HEADER_FOOTER_MARGIN = 0.1  # top/bottom 10% of page
HEADING_FONT_SIZE = 12  # consider text >12pt as heading
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_image_hash(pix):
    """Return md5 hash of image bytes for deduplication."""
    img_bytes = pix.tobytes("png")
    return hashlib.md5(img_bytes).hexdigest()

def get_headings_near_image(image_bbox, text_blocks):
    """Find the nearest heading above the image based on font size."""
    img_top = image_bbox.y0
    headings = []
    for block in text_blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font_size = span["size"]
                text = span["text"].strip()
                if not text:
                    continue
                # consider big text as heading
                if font_size >= HEADING_FONT_SIZE:
                    y_pos = span["bbox"][1]
                    # heading must be above the image
                    if y_pos < img_top:
                        headings.append((y_pos, text))
    if headings:
        # closest heading above image
        return sorted(headings, key=lambda h: abs(img_top - h[0]))[0][1]
    return None

def extract_images_with_headings(pdf_path):
    doc = fitz.open(pdf_path)
    seen_hashes = set()
    results = []

    for page_num, page in enumerate(doc, start=1):
        print(f"Processing page {page_num}...")
        text_blocks = page.get_text("dict")["blocks"]
        images = page.get_images(full=True)
        page_height = page.rect.height

        for img_index, img in enumerate(images, start=1):
            xref = img[0]
            bbox = page.get_image_bbox(img)
            pix = fitz.Pixmap(doc, xref)
            ihash = get_image_hash(pix)

            # --- Filtering header/footer/logo ---
            if pix.width < MIN_IMAGE_SIZE or pix.height < MIN_IMAGE_SIZE:
                continue
            if bbox.y0 < page_height * HEADER_FOOTER_MARGIN or bbox.y1 > page_height * (1 - HEADER_FOOTER_MARGIN):
                continue
            if ihash in seen_hashes:
                continue
            seen_hashes.add(ihash)
            # ------------------------------------

            heading = get_headings_near_image(bbox, text_blocks) or "NoHeading"
            heading_clean = heading[:30].replace(" ", "_").replace("/", "_")

            filename = f"page{page_num}_img{img_index}_{heading_clean}.png"
            out_path = os.path.join(OUTPUT_DIR, filename)

            pix.save(out_path)
            print(f"✅ Saved: {out_path} | Heading: {heading}")
            results.append({
                "page": page_num,
                "file": out_path,
                "heading": heading
            })

            pix = None  # free memory

    doc.close()
    print("\nExtraction complete!")
    return results

if __name__ == "__main__":
    results = extract_images_with_headings(PDF_PATH)
    print("\nSummary:")
    for r in results:
        print(f"Page {r['page']}: {os.path.basename(r['file'])']} under heading '{r['heading']}'")
