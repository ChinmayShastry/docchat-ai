# TEXT EXTRACTION
# Converts uploaded files into LangChain Document objects.
# Each Document carries page_content (text) and metadata (source, page number).
# Supports: PDF, DOCX, TXT, XLSX/XLS, CSV

from langchain_core.documents import Document
import docx, openpyxl, pandas as pd
from config import Config
import os

def extract_text(filepath, filename):
    ext  = filename.split(".")[-1].lower()
    docs = []

    if ext == "pdf":
        # Primary: native PDF text extraction via PyPDFLoader
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(filepath)
            pages  = loader.load()

            if any(p.page_content.strip() for p in pages):
                for p in pages:
                    p.metadata["source"]   = filename
                    p.metadata["doc_name"] = filename
                return pages

        except Exception as e:
            print(f"[PDF native extraction failed] {e}")

        # Fallback: OCR via Tesseract (requires tesseract + poppler installed)
        # If these binaries are missing this will also fail — the caller checks
        # for an empty return and surfaces an error to the user.
        try:
            import pytesseract
            from pdf2image import convert_from_path
            images = convert_from_path(filepath, dpi=200)
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename, "page": i + 1, "doc_name": filename}
                ))
        except Exception as e:
            print(f"[PDF OCR fallback failed] {e}")

    elif ext == "docx":
        doc  = docx.Document(filepath)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        docs.append(Document(
            page_content=text,
            metadata={"source": filename, "page": 1, "doc_name": filename}
        ))

    elif ext == "txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs.append(Document(
            page_content=text,
            metadata={"source": filename, "page": 1, "doc_name": filename}
        ))

    elif ext in ["xlsx", "xls"]:
        wb   = openpyxl.load_workbook(filepath, data_only=True)
        text = ""
        for sheet in wb.sheetnames:
            ws    = wb[sheet]
            text += f"\n[Sheet: {sheet}]\n"
            for row in ws.iter_rows(values_only=True):
                row_text = " | ".join([str(c) for c in row if c is not None])
                if row_text.strip():
                    text += row_text + "\n"
        docs.append(Document(
            page_content=text,
            metadata={"source": filename, "page": 1, "doc_name": filename}
        ))

    elif ext == "csv":
        df   = pd.read_csv(filepath)
        # Include schema info + a 100-row sample so large CSVs don't overflow context
        text = (
            f"Dataset: {filename}\n"
            f"Rows: {len(df)}\n"
            f"Columns: {', '.join(df.columns.tolist())}\n\n"
        )
        sample_df = df.sample(min(len(df), 100), random_state=42)
        text += sample_df.to_string()
        docs.append(Document(
            page_content=text,
            metadata={"source": filename, "page": 1, "doc_name": filename}
        ))

    return docs