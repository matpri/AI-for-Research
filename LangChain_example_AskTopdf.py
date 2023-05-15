# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:31:02 2023

@author: MPrina
"""
#https://www.youtube.com/watch?v=TLf90ipMzfE

from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

# Get your API keys from openai, you will need to create an account. 
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
import os
os.environ["OPENAI_API_KEY"] = ""

# location of the pdf file/files. 
reader = PdfReader(r"paper_near_optimal.pdf")

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text
        
# print(raw_text[:1000])

# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

print(len(texts))
print('--------------------')

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_texts(texts, embeddings)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(), chain_type="stuff")

# query = "how many optimal and near-optimal solutions are found in the article?"
# docs = docsearch.similarity_search(query)
# result= chain.run(input_documents=docs, question=query)
# print(result)

# query = "what is the sequential order of clusters from low to high CO2 emissions reduction in the results of the article?"
# docs = docsearch.similarity_search(query)
# result= chain.run(input_documents=docs, question=query)
# print(result)

# query = "what is the maximum rooftop PV potential considered as assumption in the study?"
# docs = docsearch.similarity_search(query)
# result= chain.run(input_documents=docs, question=query)
# print(result)

query = "what is the optimal number of clusters for the considered problem?"
docs = docsearch.similarity_search(query)
result= chain.run(input_documents=docs, question=query)
print(result)