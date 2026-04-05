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

    def process_directory(self, directory_path):
        """Processes all PDFs and returns a tuple: (list_of_chunks, list_of_metadatas)."""
        all_chunks = []
        all_metadatas = []

        if not os.path.exists(directory_path):
            print(f"Error: Directory {directory_path} not found.")
            return [], []

        for filename in os.listdir(directory_path):
            if filename.endswith(".pdf"):
                print(f"Processing: {filename}")
                path = os.path.join(directory_path, filename)
                
                try:
                    reader = pypdf.PdfReader(path)
                    for i, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if not page_text:
                            continue
                        
                        # Create chunks for this specific page
                        page_chunks = self.splitter.split_text(page_text)
                        
                        for chunk in page_chunks:
                            all_chunks.append(chunk)
                            # Enhanced metadata for citations and filtering
                            all_metadatas.append({
                                "source": filename,
                                "page": i + 1,
                                "char_count": len(chunk)
                            })
                except Exception as e:
                    print(f"Could not process {filename}: {e}")
                        
        return all_chunks, all_metadatas
