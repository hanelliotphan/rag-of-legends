# ---------------------------------------------------------------------------- #
# Author: Han-Elliot Phan                                                      #
# Email: hanelliotphan@gmail.com                                               #
#                                                                              #
# Last update: February 25, 2025                                               #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# langchain.py                                                                 #
#                                                                              #
# This file is for the setup of the LangChain pipeline of the RAG of Legends   #
# project.                                                                     #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                               IMPORT LIBRARIES                               #
# ---------------------------------------------------------------------------- #

import glob
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory


def collect_knowledge(folder_path):
    """
    collect_knowledge -- Collect documents of knowledge base
    """
    folders = glob.glob(folder_path)
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader)
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata['doc_type'] = doc_type
            documents.append(doc)
    return documents


def create_chroma_vector_store(doc_chunks, embeddings, db_name):
    """
    create_chroma_vector_store -- Create Chroma vector store
    """
    return Chroma.from_documents(
        documents=doc_chunks,
        embedding=embeddings,
        persist_directory=db_name
    )


def create_conversation_chain(vector_store):
    """
    create_conversation_chain -- Create conversation chain with OpenAI's GPT-4o 
    mini model
    """
    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    retriever = vector_store.as_retriever()
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )