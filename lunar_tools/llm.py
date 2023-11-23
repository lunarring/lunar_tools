#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import threading
import time
from openai import OpenAI
import os
from lunar_tools.logprint import LogPrint



class GPT4:
    def __init__(
        self, 
        client=None, 
        logger=None, 
        model="gpt-4-0613",
    ):
        """
        super simple GPT4 wrapper. docstring coming soon
        """
        if client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No OPENAI_API_KEY found in environment variables")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = client
        self.logger = logger if logger else LogPrint()
        self.model = model

    def generate(self, prompt):
        """
        docstring coming soon
        """
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
        )
        return chat_completion.choices[0].message.content




#%% EXAMPLE USE        
if __name__ == "__main__":
    gpt4 = GPT4()
    msg = gpt4.generate("tell me about yourself")
    # player = SoundPlayer()
    # player.play_sound("/tmp/bla.mp3")
    # player.stop_sound()
