from langchain_core.documents import Document
import docx
import openpyxl
import pandas as pd


def extract_text(filepath, filename):
    ext = filename.split(".")[-1].lower()
    docs = []

    # ── PDF ─────────────────────────────────────────
    if ext == "pdf":
        try:
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(filepath)
            pages = loader.load()

            # Remove empty pages
            pages = [
                p for p in pages
                if p.page_content and p.page_content.strip()
            ]

            if pages:
                for p in pages:
                    p.metadata["source"] = filename
                    p.metadata["doc_name"] = filename
                return pages

        except Exception as e:
            print(f"[PDF native extraction failed] {e}")

        # OCR fallback
        try:
            import pytesseract
            from pdf2image import convert_from_path

            images = convert_from_path(filepath, dpi=200)

            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)

                if text and text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": i + 1,
                            "doc_name": filename
                        }
                    ))

        except Exception as e:
            print(f"[PDF OCR fallback failed] {e}")

    # ── DOCX ────────────────────────────────────────
    elif ext == "docx":
        doc = docx.Document(filepath)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "page": 1, "doc_name": filename}
            ))

    # ── TXT ─────────────────────────────────────────
    elif ext == "txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "page": 1, "doc_name": filename}
            ))

    # ── Excel ───────────────────────────────────────
    elif ext in ["xlsx", "xls"]:
        wb = openpyxl.load_workbook(filepath, data_only=True)
        text = ""

        for sheet in wb.sheetnames:
            ws = wb[sheet]
            text += f"\n[Sheet: {sheet}]\n"

            for row in ws.iter_rows(values_only=True):
                row_text = " | ".join([str(c) for c in row if c is not None])
                if row_text.strip():
                    text += row_text + "\n"

        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "page": 1, "doc_name": filename}
            ))

    # ── CSV ─────────────────────────────────────────
    elif ext == "csv":
        try:
            df = pd.read_csv(filepath, on_bad_lines="skip")
        except Exception as e:
            print(f"[CSV read error] {e}")
            return docs

        text = (
            f"Dataset: {filename}\n"
            f"Rows: {len(df)}\n"
            f"Columns: {', '.join(df.columns.tolist())}\n\n"
        )

        sample_df = df.sample(min(len(df), 100), random_state=42)
        text += sample_df.to_string()

        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "page": 1, "doc_name": filename}
            ))

    return docs
