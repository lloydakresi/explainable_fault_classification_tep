import pymupdf.layout  # activate PyMuPDF-Layout in pymupdf
import pymupdf4llm
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

pdf_folder = Path("knowledge base")
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("tep_knowledge")
global_id = 0

for pdf_file in pdf_folder.glob("*.pdf"):

    # The remainder of the script is unchanged
    md_text = pymupdf4llm.to_markdown(
        pdf_file,
        write_images=True,
        footer=False,
        header=False,
        page_chunks=True,
        image_path="images"
    )

    chunks = []

    for page in md_text:
        page_num = page["metadata"]["page_number"]
        text = page["text"]
        paragraphs = text.split("\n\n")

        for p in paragraphs:
            p = p.strip()
            if p.startswith("- ["):
                continue
            if p.startswith("- ")
            if len(p) > 100:
                chunks.append((p, page_num))


    for chunk, page_num in chunks:

        embedding = model.encode(chunk).tolist()

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(global_id)],
            metadatas=[{
                "source": str(pdf_file),
                "page": page_num
            }]
        )

        global_id += 1

query = "What is the Tennessee Eastman process?"

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print(results["documents"])
