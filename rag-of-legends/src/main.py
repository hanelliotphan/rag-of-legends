# ---------------------------------------------------------------------------- #
# Author: Han-Elliot Phan                                                      #
# Email: hanelliotphan@gmail.com                                               #
#                                                                              #
# Last update: March 3, 2025                                                   #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# main.py                                                                      #
#                                                                              #
# This file is for the main execution of the RAG of Legends project.           #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                               IMPORT LIBRARIES                               #
# ---------------------------------------------------------------------------- #

import argparse
import gradio as gr
import os
import sys
from model.langchain import (
    collect_knowledge,
    collect_knowledge_document_chunks,
    get_openai_embeddings,
    clear_chroma_vector_store,
    create_chroma_vector_store,
    create_conversation_chain
)
from utils.chat import (
    get_chat_response,
)
from utils.login import (
    hf_login,
    openai_login
)


# ---------------------------------------------------------------------------- #
#                               MAIN STREAMLINE                                #
# ---------------------------------------------------------------------------- #

def main_streamline(
    folder_path,
    db_name,
    message
):
    documents = collect_knowledge(folder_path=folder_path)
    doc_chunks = collect_knowledge_document_chunks(documents=documents)
    embeddings = get_openai_embeddings()
    clear_chroma_vector_store(embeddings=embeddings, db_name=db_name)
    vector_store = create_chroma_vector_store(
        doc_chunks=doc_chunks,
        embeddings=embeddings,
        db_name=db_name
    )
    conv_chain = create_conversation_chain(vector_store=vector_store)
    gr.ChatInterface(
        get_chat_response(
            conversation_chain=conv_chain,
            message=message,
            history=None
        )
    ).launch()


def main():
    main_streamline(
        folder_path="./knowledge-base/*",
        db_name="vector_db",
        message="Who is Caitlyn?"
    )

# ---------------------------------------------------------------------------- #
#                               MAIN EXECUTION                                 #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()