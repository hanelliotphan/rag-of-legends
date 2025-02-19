# ---------------------------------------------------------------------------- #
# Author: Han-Elliot Phan                                                      #
# Email: hanelliotphan@gmail.com                                               #
#                                                                              #
# Last update: February 18, 2025                                               #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# login.py                                                                     #
#                                                                              #
# This file is for setting up Hugging Face & OpenAI login using HF_TOKEN and   #
# OPENAI_API_KEY environment variables.                                        #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                               IMPORT LIBRARIES                               #
# ---------------------------------------------------------------------------- #

import logging
import huggingface_hub
import sys

from openai import OpenAI


# ---------------------------------------------------------------------------- #
#                                LOGIN FUNCTIONS                               #
# ---------------------------------------------------------------------------- #

def hf_login(token):
    """
    hf_login -- Log in to Hugging Face interface using Hugging Face access token

    Documentation: https://huggingface.co/docs/huggingface_hub/en/package_reference/authentication#huggingface_hub.login
    """
    try:
        huggingface_hub.login(token=token)
        logging.info(msg=f"[login.py] hf_login: Successfully logged in to Hugging Face.")
    except Exception as e:
        logging.critical(msg=f"[login.py] hf_login: Cannot log in to Hugging Face. Error message: {e}")
        sys.exit(-1)


def openai_login(api_key):
    """
    openai_login -- Log in to OpenAI interface using OpenAI API key

    Documentation: https://platform.openai.com/docs/quickstart
    """
    openai_client = OpenAI(api_key=api_key)
    try:
        openai_client.models.list()
        logging.info(msg="[login.py] openai_login: Successfully logged in to OpenAI")
    except Exception as e:
        logging.critical(msg=f"[login.py] openai_login: Cannot log in to OpenAI. Error message: {e}")
        sys.exit(-1)
    return openai_client