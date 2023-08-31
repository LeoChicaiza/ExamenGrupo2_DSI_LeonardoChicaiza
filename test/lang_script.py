import os

from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import streamlit as st



os.environ['OPENAI_API_KEY'] = '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'
default_doc_name = 'documento.pdf'

def process_doc(
        path: str = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf',
        is_local: bool = False,
        question: str = 'Cual es el nombre del pdf?'
):
    _, loader = os.system(f'curl -o {default_doc_name} {path}'), PyPDFLoader(f"./{default_doc_name}") if not is_local \
        else PyPDFLoader(path)

    documento = loader.load_and_split()

    print(documento[-1])

    db = FAISS.from_documents(documento, embedding=OpenAIEmbeddings())

    pr = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='map_reduce', retriever=db.as_retriever())

    st.write(pr.run(question))
    print(pr.run(question))

def client():
    st.title('Manage LLM with LangChain')
    uploader = st.file_uploader('Upload PDF', type='pdf')

    if uploader:
        with open(f'./{default_doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('PDF saved!!')

    question = st.text_input('Genera un resumen de 20 palabras sobre el pdf',
                   placeholder='Give response about your PDF', disabled=not uploader)

    if st.button('Send Question'):
        if uploader:
            process_doc(
                path=default_doc_name,
                is_local=True,
                question=question
            )
        else:
            st.info('Loading default PDF')
            process_doc()

if __name__ == '__main__':
        client()