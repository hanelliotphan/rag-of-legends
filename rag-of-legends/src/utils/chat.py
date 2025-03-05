# ---------------------------------------------------------------------------- #
# Author: Han-Elliot Phan                                                      #
# Email: hanelliotphan@gmail.com                                               #
#                                                                              #
# Last update: March 4, 2025                                                   #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# chat.py                                                                      #
#                                                                              #
# This file is for setting up Gradio chat interface for the RAG of Legends AI  #
# knowledge agent                                                              #
# ---------------------------------------------------------------------------- #


def get_chat_response(conversation_chain, message, history):
    """
    get_chat_response -- Set up chat response fetching
    """
    result = conversation_chain.invoke({"question": message})
    response = result["answer"]
    return response