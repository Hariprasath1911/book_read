from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
import streamlit as st
import pickle
with open("raw.pkl", "rb") as file:
    raw = pickle.load(file)

text_spliter=CharacterTextSplitter(separator="\n",chunk_size=1000,chunk_overlap=198)
text=text_spliter.split_text(raw)

openaikey=""

doc=FAISS.from_texts(text,OpenAIEmbeddings(openai_api_key=openaikey))

llm=OpenAI(openai_api_key=openaikey)

chain=load_qa_chain(llm,chain_type="map_reduce")

st.title("Tender Document Q&A")
query = st.text_input("Enter your query:")

if st.button('Run'):
    output=doc.similarity_search(query)
    a=chain.run(input_documents=output,question=query)
    st.write(a)
