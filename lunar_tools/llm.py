#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import threading
import time
from openai import OpenAI
import os
from lunar_tools.logprint import LogPrint
from lunar_tools.utils import read_api_key

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None



class OpenAIWrapper:
    """
    A simple wrapper class for the GPT models provided by OpenAI.
    
    Attributes:
        client (OpenAI): An instance of the OpenAI API client.
        logger: A logger instance for logging messages. Defaults to a basic logger.
        model (str): The identifier of the GPT model to be used.
        available_models (list): A list of available GPT model identifiers.
    """

    def __init__(self, client=None, logger=None, model="gpt-4-0613"):
        """
        Initializes the OpenAIWrapper class with a client, logger, and model.

        Args:
            client: An instance of the OpenAI API client. If None, it will initialize with the API key from environment variables.
            logger: A logging instance. If None, a basic logger will be used.
            model (str): The initial model to use. Defaults to "gpt-4-0613".
        """
        if client is None:
            api_key = read_api_key('OPENAI_API_KEY') 
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = client

        self.logger = logger if logger else LogPrint()
        self.model = model
        self.available_models = ["gpt-4-1106-preview", "gpt-4-0613", "gpt-3.5-turbo-1106"]

    def list_available_models(self):
        """
        Lists the available GPT models.

        Returns:
            list: A list of available GPT model identifiers.
        """
        return self.available_models

    def set_model(self, model_name):
        """
        Sets the model to be used, if it is available in the list of models.

        Args:
            model_name (str): The model identifier to set.

        Raises:
            ValueError: If the model_name is not in the list of available models.
        """
        if model_name in self.available_models:
            self.model = model_name
        else:
            raise ValueError(f"Model {model_name} is not available.")

    def generate(self, prompt):
        """
        Generates a response based on the given prompt using the selected GPT model.

        Args:
            prompt (str): The input prompt to generate a response for.

        Returns:
            str: The generated response from the GPT model.
        """
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt},
            ],
            model=self.model,
        )
        return chat_completion.choices[0].message.content


class Gemini:
    """
    A simple wrapper class for the Gemini models provided by Google.
    
    Attributes:
        client: An instance of the Google GenAI client.
        logger: A logger instance for logging messages. Defaults to a basic logger.
        model (str): The identifier of the Gemini model to be used.
        available_models (list): A list of available Gemini model identifiers.
    """

    def __init__(self, client=None, logger=None, model="gemini-2.0-flash-exp"):
        """
        Initializes the Gemini class with a client, logger, and model.

        Args:
            client: An instance of the Google GenAI client. If None, it will initialize with the API key from environment variables.
            logger: A logging instance. If None, a basic logger will be used.
            model (str): The initial model to use. Defaults to "gemini-2.0-flash-exp".
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai package is not installed. Install it with: pip install google-genai")
        
        if client is None:
            api_key = read_api_key('GEMINI_API_KEY')
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = client

        self.logger = logger if logger else LogPrint()
        self.model = model
        self.available_models = [
            "gemini-2.0-flash-exp", 
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-pro"
        ]

    def list_available_models(self):
        """
        Lists the available Gemini models.

        Returns:
            list: A list of available Gemini model identifiers.
        """
        return self.available_models

    def set_model(self, model_name):
        """
        Sets the model to be used, if it is available in the list of models.

        Args:
            model_name (str): The model identifier to set.

        Raises:
            ValueError: If the model_name is not in the list of available models.
        """
        if model_name in self.available_models:
            self.model = model_name
        else:
            raise ValueError(f"Model {model_name} is not available.")

    def generate(self, prompt):
        """
        Generates a response based on the given prompt using the selected Gemini model.

        Args:
            prompt (str): The input prompt to generate a response for.

        Returns:
            str: The generated response from the Gemini model.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text


class Deepseek:
    """
    A simple wrapper class for the DeepSeek models provided by DeepSeek.
    
    Attributes:
        client (OpenAI): An instance of the OpenAI API client configured for DeepSeek.
        logger: A logger instance for logging messages. Defaults to a basic logger.
        model (str): The identifier of the DeepSeek model to be used.
        available_models (list): A list of available DeepSeek model identifiers.
    """

    def __init__(self, client=None, logger=None, model="deepseek-chat"):
        """
        Initializes the Deepseek class with a client, logger, and model.

        Args:
            client: An instance of the OpenAI API client configured for DeepSeek. If None, it will initialize with the API key from environment variables.
            logger: A logging instance. If None, a basic logger will be used.
            model (str): The initial model to use. Defaults to "deepseek-chat".
        """
        if client is None:
            api_key = read_api_key('DEEPSEEK_API_KEY')
            self.client = OpenAI(
                api_key=api_key, 
                base_url="https://api.deepseek.com"
            )
        else:
            self.client = client

        self.logger = logger if logger else LogPrint()
        self.model = model
        self.available_models = ["deepseek-chat", "deepseek-reasoner"]

    def list_available_models(self):
        """
        Lists the available DeepSeek models.

        Returns:
            list: A list of available DeepSeek model identifiers.
        """
        return self.available_models

    def set_model(self, model_name):
        """
        Sets the model to be used, if it is available in the list of models.

        Args:
            model_name (str): The model identifier to set.

        Raises:
            ValueError: If the model_name is not in the list of available models.
        """
        if model_name in self.available_models:
            self.model = model_name
        else:
            raise ValueError(f"Model {model_name} is not available.")

    def generate(self, prompt):
        """
        Generates a response based on the given prompt using the selected DeepSeek model.

        Args:
            prompt (str): The input prompt to generate a response for.

        Returns:
            str: The generated response from the DeepSeek model.
        """
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt},
            ],
            model=self.model,
        )
        return chat_completion.choices[0].message.content




#%% EXAMPLE USE        
if __name__ == "__main__":
    # OpenAIWrapper Example
    print("Testing OpenAIWrapper...")
    openai = OpenAIWrapper(model="gpt-5-2025-08-07")
    gpt_msg = openai.generate("tell me about yourself in one sentence. which model are you?")
    print(f"OpenAI Response: {gpt_msg}")
    
    # Gemini Example
    print("\nTesting Gemini...")
    try:
        gemini = Gemini()
        gemini_msg = gemini.generate("tell me about yourself in one sentence")
        print(f"Gemini Response: {gemini_msg}")
    except ImportError as e:
        print(f"Gemini not available: {e}")
    except Exception as e:
        print(f"Gemini error: {e}")

    # DeepSeek Example
    print("\nTesting DeepSeek...")
    try:
        deepseek = Deepseek()
        deepseek_msg = deepseek.generate("tell me about yourself in one sentence")
        print(f"DeepSeek Response: {deepseek_msg}")
    except Exception as e:
        print(f"DeepSeek error: {e}")
