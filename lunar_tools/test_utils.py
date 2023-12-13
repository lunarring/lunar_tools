#%%

from utils import *
# %%


get_config_path()

# %%


save_api_key_to_lunar_config("openai", "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")
# %%

read_api_key('openai')

# %%

import lunar_tools as lt
gpt4 = lt.GPT4()
msg = gpt4.generate("tell me about yourself")
# %%
