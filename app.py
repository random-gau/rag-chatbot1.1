import streamlit as st
from dotenv import load_dotenv

# Loads GROQ_API_KEY from a local .env file. On HF Spaces, secrets are
# injected as real environment variables, so this call is a no-op there
# and no code path changes between local and deployed.
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "faiss_index"
RETRIEVER_K = 4

PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "If the context does not contain the answer, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {input}"
)


@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(
        INDEX_DIR, embedding_model, allow_dangerous_deserialization=True
    )


@st.cache_resource
def load_chain():
    # GROQ_API_KEY is read implicitly from the environment by ChatGroq.
    # On Streamlit Community Cloud / HF Spaces, set it as a secret, not here.
    # openai/gpt-oss-20b is Groq's current free/developer-tier text model
    # (checked live against console.groq.com/docs/models on 2026-09-14 --
    # llama-3.1-8b-instant, which older tutorials reference, was moved to
    # Enterprise-only in June 2026 and will 4xx on a free key). Swap to
    # openai/gpt-oss-120b if answer quality on your eval set (step 3) needs it.
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    retriever = load_vectorstore().as_retriever(search_kwargs={"k": RETRIEVER_K})
    combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
    return create_retrieval_chain(retriever, combine_docs_chain)


def main():
    st.title("RAG Chatbot")

    chain = load_chain()
    user_question = st.text_input("Ask a question about the documents:")

    if user_question:
        with st.spinner("Thinking..."):
            result = chain.invoke({"input": user_question})

        st.markdown(f"**Answer:** {result['answer']}")

        with st.expander("Retrieved context"):
            for i, doc in enumerate(result["context"], start=1):
                st.markdown(f"**Chunk {i}** (source: {doc.metadata.get('source', '?')}, "
                            f"page {doc.metadata.get('page', '?')})")
                st.text(doc.page_content)


if __name__ == "__main__":
    main()