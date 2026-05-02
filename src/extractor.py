# ── PDF ─────────────────────────────────────────
    # if ext == "pdf":
    # docs = []
    # print("DEBUG: Starting PDF extraction")
    
    # # Try PyPDFLoader first
    # try:
    #     from langchain_community.document_loaders import PyPDFLoader

    #     loader = PyPDFLoader(filepath)
    #     pages = loader.load()

    #     pages = [
    #         p for p in pages
    #         if p.page_content and p.page_content.strip()
    #     ]

    #     if pages:
    #         for p in pages:
    #             p.metadata["source"] = filename
    #             p.metadata["doc_name"] = filename

    #         print("✅ PDF parsed using PyPDFLoader")
    #         return pages

    # except Exception as e:
    #     print(f"[PyPDFLoader failed] {e}")
    #     print("DEBUG: PyPDFLoader pages:", len(pages))

    # # 🔥 Fallback: PyMuPDF (VERY IMPORTANT)
    # try:
    #     import fitz  # PyMuPDF

    #     doc = fitz.open(filepath)

    #     for i, page in enumerate(doc):
    #         text = page.get_text()

    #         if text and text.strip():
    #             docs.append(Document(
    #                 page_content=text,
    #                 metadata={
    #                     "source": filename,
    #                     "page": i + 1,
    #                     "doc_name": filename
    #                 }
    #             ))

    #     if docs:
    #         print("✅ PDF parsed using PyMuPDF")
    #         return docs

    # except Exception as e:
    #     print(f"[PyMuPDF failed] {e}")
    #     print("DEBUG: PyMuPDF docs:", len(docs))



from langchain_core.documents import Document
import docx
import openpyxl
import pandas as pd


def extract_text(filepath, filename):
    ext = filename.split(".")[-1].lower()
    docs = []

    # ── PDF ─────────────────────────────────────────
    if ext == "pdf":
        print("DEBUG: Starting PDF extraction")

        # ── Try PyPDFLoader ─────────────────────────
        try:
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(filepath)
            pages = loader.load()

            pages = [
                p for p in pages
                if p.page_content and p.page_content.strip()
            ]

            print(f"DEBUG: PyPDFLoader pages after filter: {len(pages)}")

            if pages:
                for p in pages:
                    p.metadata["source"] = filename
                    p.metadata["doc_name"] = filename

                print("✅ PDF parsed using PyPDFLoader")
                return pages

        except Exception as e:
            print(f"[PyPDFLoader failed] {e}")

        # ── Fallback: PyMuPDF ─────────────────────────
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(filepath)

            for i, page in enumerate(doc):
                text = page.get_text()

                if text and text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": i + 1,
                            "doc_name": filename
                        }
                    ))

            print(f"DEBUG: PyMuPDF docs after filter: {len(docs)}")

            if docs:
                print("✅ PDF parsed using PyMuPDF")
                return docs

        except Exception as e:
            print(f"[PyMuPDF failed] {e}")

        # ── OCR fallback ─────────────────────────
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

        if text and text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "page": 1, "doc_name": filename}
            ))

    # ── TXT ─────────────────────────────────────────
    elif ext == "txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        if text and text.strip():
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

        if text and text.strip():
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

        if text and text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "page": 1, "doc_name": filename}
            ))

    return docs
