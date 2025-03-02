# ---------------------------------------------------------------------------- #
# Author: Han-Elliot Phan                                                      #
# Email: hanelliotphan@gmail.com                                               #
#                                                                              #
# Last update: March 1, 2025                                                   #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# chat.py                                                                      #
#                                                                              #
# This file is for setting up Gradio chat interface for the RAG of Legends AI  #
# knowledge agent                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                               IMPORT LIBRARIES                               #
# ---------------------------------------------------------------------------- #

import gradio as gr


# ---------------------------------------------------------------------------- #
#                               IMPORT LIBRARIES                               #
# ---------------------------------------------------------------------------- #

def get_chat_response(conversation_chain, message, history):
    """
    get_chat_response -- Set up chat response fetching
    """
    result = conversation_chain.invoke({"question": message})
    response = result["answer"]
    return response


def get_gradio_chat_interface(chat):
    """
    get_gradio_chat_interface -- Get Gradio chat interface
    """
    return gr.ChatInterface(chat).launch()