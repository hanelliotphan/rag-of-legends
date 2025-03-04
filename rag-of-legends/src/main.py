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
    get_gradio_chat_interface
)

from utils.login import (
    hf_login,
    openai_login
)


# ---------------------------------------------------------------------------- #
#                               MAIN STREAMLINE                                #
# ---------------------------------------------------------------------------- #

def main_streamline():
    pass
    # TODO


def main():
    pass
    # TODO

# ---------------------------------------------------------------------------- #
#                               MAIN EXECUTION                                 #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()