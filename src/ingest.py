import pypdf
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RailDocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def extract_text_from_pdf(self, pdf_path):
        """Extracts all text from a given PDF path."""
        reader = pypdf.PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text
        return full_text

    def create_chunks(self, text):
        """Splits text into manageable chunks for embedding."""
        return self.splitter.split_text(text)

    def process_directory(self, directory_path):
        """Processes all PDFs in a directory and returns a list of chunks."""
        all_chunks = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".pdf"):
                print(f"Processing: {filename}")
                path = os.path.join(directory_path, filename)
                text = self.extract_text_from_pdf(path)
                chunks = self.create_chunks(text)
                all_chunks.extend(chunks)
        return all_chunks
