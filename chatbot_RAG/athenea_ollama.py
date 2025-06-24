import streamlit as st
import requests

# URL del servicio de embeddings
EMBEDDINGS_SERVICE_URL = "http://localhost:8001/retrieve"

# URL de Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"

# Función para recuperar documentos del servicio de embeddings
def retrieve_documents(query, k=2):
    try:
        response = requests.post(EMBEDDINGS_SERVICE_URL, json={"query": query, "k": k})
        return response.json()["documents"]
    except Exception as e:
        st.error(f"Error al recuperar documentos: {str(e)}")
        return []

# Función para generar respuesta usando Ollama
def generate_response(query, relevant_docs):
    prompt = f"Consulta: {query}\nDocumentos relevantes: {' '.join(relevant_docs)}\nRespuesta:"
    payload = {
        "model": "llama3:8b",  # indicamos el modelo
        "prompt": prompt
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return response.json()["response"]
    except Exception as e:
        return f"Error al conectar con Ollama: {str(e)}"

# Interfaz de Streamlit
st.title("Athenea - Demo")

query = st.text_input("Ingrese su consulta:")
if query:
    with st.spinner("Buscando documentos y generando respuesta..."):
        relevant_docs = retrieve_documents(query)
        if relevant_docs:
            st.write("**Documentos relevantes encontrados:**")
            for doc in relevant_docs:
                st.write(f"- {doc[:200]}...")  # Mostrar solo un fragmento
            response = generate_response(query, relevant_docs)
            st.write("**Respuesta del chatbot:**")
            st.write(response)
        else:
            st.write("No se encontraron documentos relevantes.")