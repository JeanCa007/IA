import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Cargar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L12-v2')

# Carpeta con documentos
DOCS_FOLDER = "docs"

# Función para cargar documentos desde la carpeta
def load_documents():
    documents = []
    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(DOCS_FOLDER, filename), "r", encoding="utf-8") as f:
                documents.append(f.read())
    return documents

# Generar y almacenar embeddings
def generate_embeddings():
    documents = load_documents()
    if not documents:
        raise ValueError("No se encontraron documentos en la carpeta 'docs'.")
    embeddings = model.encode(documents, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, "embeddings.index")
    np.save("documents.npy", documents)
    return documents, index

# Cargar embeddings y documentos si ya existen, o generarlos
if os.path.exists("embeddings.index") and os.path.exists("documents.npy"):
    index = faiss.read_index("embeddings.index")
    documents = np.load("documents.npy", allow_pickle=True)
else:
    documents, index = generate_embeddings()

# API con FastAPI
app = FastAPI()

class Query(BaseModel):
    query: str
    k: int = 2

@app.post("/retrieve")
def retrieve_documents(query: Query):
    try:
        query_embedding = model.encode([query.query])
        distances, indices = index.search(query_embedding, query.k)
        relevant_docs = [documents[i] for i in indices[0]]
        return {"documents": relevant_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ejecutar el servicio
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)