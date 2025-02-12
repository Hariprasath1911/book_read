from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
import streamlit as st
tender=PdfReader("C:/Users/DINESH/Desktop/Data for DS/vscode/openai/tender.pdf")

raw=""
for i,j in enumerate(tender.pages):
    if j.extract_text():
        raw=raw+j.extract_text()

raw=raw.replace("\n","")

text_spliter=CharacterTextSplitter(separator="\n",chunk_size=1000,chunk_overlap=198)
text=text_spliter.split_text(raw)

openaikey="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"

doc=FAISS.from_texts(text,OpenAIEmbeddings(openai_api_key=openaikey))

llm=OpenAI(openai_api_key=openaikey)

chain=load_qa_chain(llm,chain_type="map_reduce")

st.title("Tender Document Q&A")
query = st.text_input("Enter your query:")

if st.button('Run'):
    output=doc.similarity_search(query)
    a=chain.run(input_documents=output,question=query)
    st.write(a)