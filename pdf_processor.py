import fitz
import numpy as np
from PIL import Image
from utils import normalize_bbox


def extract_text_and_bbox_from_pdf(pdf_path):
    """
    Extracts text and bounding boxes from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a page in the PDF.
              Each dictionary contains the following keys:
              - 'image': The image of the page in RGB format.
              - 'text': A list of dictionaries, where each dictionary represents a text block.
                        Each text block dictionary contains the following keys:
                        - 'text': The text content of the block.
                        - 'bbox': The bounding box coordinates of the block.
    """
    pdf_document = fitz.open(pdf_path)
    data = []
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_cv = np.array(img)
        blocks = page.get_text("dict")["blocks"]
        page_data = {'image': img_cv, 'text': []}

        if len(blocks) == 0:
            continue
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                bbox = line["bbox"]
                bbox = normalize_bbox(bbox, img_cv.shape[1], img_cv.shape[0])
                text = " ".join([span["text"] for span in line["spans"]])
                page_data['text'].append({
                    "text": text,
                    "bbox": bbox
                })
        data.append(page_data)
    pdf_document.close()

    return data


if __name__ == "__main__":
    pdf_path = "path_to_pdf_file"
    data = extract_text_and_bbox_from_pdf(pdf_path)
    for item in data:
        print(item)
