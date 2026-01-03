import sys
import os
from pypdf import PdfReader
from docx import Document

def extract_text(file_path):
    print(f"--- Processing: {os.path.basename(file_path)} ---")
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        elif ext == '.docx':
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        else:
            return f"Unsupported format: {ext}"
    except Exception as e:
        return f"Error reading {file_path}: {e}"

if __name__ == "__main__":
    for path in sys.argv[1:]:
        content = extract_text(path)
        try:
            print(content)
        except UnicodeEncodeError:
            print(content.encode('utf-8', errors='replace').decode('utf-8'))
        print("\n" + "="*50 + "\n")
