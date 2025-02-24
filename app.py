import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# Cambiamos esta línea
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Configuración inicial
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY no está configurada en las variables de entorno.")
    st.stop()

genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    """Extrae texto de los documentos PDF."""
    try:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        if not text.strip():
            raise ValueError("No se pudo extraer texto de los PDFs")
        return text
    except Exception as e:
        st.error(f"Error al procesar PDF: {str(e)}")
        return None

def get_text_chunks(text):
    """Divide el texto en chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        if not chunks:
            raise ValueError("No se pudieron crear chunks de texto")
        return chunks
    except Exception as e:
        st.error(f"Error al dividir el texto: {str(e)}")
        return None

def get_vector_store(text_chunks):
    """Crea y guarda el almacén de vectores."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error al crear el almacén de vectores: {str(e)}")
        return False

def get_conversational_chain():
    """Configura la cadena de conversación."""
    try:
        prompt_template = """
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error al crear la cadena de conversación: {str(e)}")
        return None

def user_input(user_question):
    """Procesa la entrada del usuario y genera una respuesta."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()

        if chain is None:
            return

        with st.spinner("Generando respuesta..."):
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            st.write("Respuesta:", response["output_text"])
    except Exception as e:
        st.error(f"Error al procesar la pregunta: {str(e)}")

def main():
    st.set_page_config(
        page_title="Chat con PDF usando Gemini",
        page_icon="📚",
        layout="wide"
    )

    st.header("Chat con PDF usando Gemini 💬")

    with st.sidebar:
        st.title("📄 Subir Documentos")
        pdf_docs = st.file_uploader(
            "Sube tus archivos PDF",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Procesar Documentos", type="primary"):
            if not pdf_docs:
                st.warning("Por favor, sube al menos un archivo PDF")
                return

            with st.spinner("Procesando documentos..."):
                # Procesar PDFs
                raw_text = get_pdf_text(pdf_docs)
                if raw_text is None:
                    return

                # Crear chunks
                text_chunks = get_text_chunks(raw_text)
                if text_chunks is None:
                    return

                # Crear vectorstore
                if get_vector_store(text_chunks):
                    st.success("¡Documentos procesados exitosamente!")
                else:
                    st.error("Error al procesar los documentos")
                    return

    # Área principal para preguntas
    st.divider()
    user_question = st.text_input(
        "Haz una pregunta sobre tus documentos",
        placeholder="¿Qué te gustaría saber?"
    )

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
