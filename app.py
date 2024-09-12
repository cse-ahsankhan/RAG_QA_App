import openai
import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction, DefaultEmbeddingFunction
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.agents import AgentExecutor, Tool,initialize_agent
from langchain.agents.types import AgentType
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.llms import OpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import VectorDBQA, RetrievalQA, LLMChain
from langchain.text_splitter import CharacterTextSplitter
import logging
import pandas as pd
import concurrent.futures
from langchain_community.document_loaders import UnstructuredURLLoader
from io import StringIO
from openai import OpenAI 
from langchain_openai import OpenAIEmbeddings


# Define OpenAPI Key
openai.api_key = os.getenv("xxx")
os.environ["OPENAI_API_KEY"] = "xxx"
OPENAI_API_KEY= "xxx"

#Creating a function to load the article from url and store in ChromaDB
def create_database(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=10)
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    database = Chroma.from_documents(texts, embeddings)
    return database

# Read Chroma DB and load Retival Client
def process_questions(database, OPENAI_API_KEY, question):
    LLM = ChatOpenAI(temperature=0,max_tokens=250,model='gpt-4o', api_key=OPENAI_API_KEY)

    Prompt_Template = """Use the following pieces of context to answer the question at the end. 
            {context}

            Question: {question}
            
            """
    PROMPT = PromptTemplate(template=Prompt_Template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    retreiver = RetrievalQA.from_chain_type(llm=LLM, chain_type="stuff", retriever=database.as_retriever(), chain_type_kwargs=chain_type_kwargs)
    answer = retreiver({"query": question})
    result = answer.get('result', '')
    return result

# Define URL
url = "URL_HERE",

#Creating a local ChromDB
database = create_database(url)

# I loaded the questions into an txt
with open("questions.txt", "r") as questions:
    lines = questions.readlines()
quests = []
for l in lines:
    l1 = l.split(",")
    quests.append(l1[0].replace("\n",""))

answers = []
i=1

# Iterating through each question and generating reponses for them
for que in quests:
    print(que)
    ans = process_questions(database,OPENAI_API_KEY,que)
    i = i+1
    with open("responses.txt", "a") as output:
        output.write(f"Response {i}")
        output.write(f"Answer: {ans}\n")
        output.write("_" * 100 + "\n")


