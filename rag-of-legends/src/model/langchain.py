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

from langchain.chains import conversational_retrieval
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory

