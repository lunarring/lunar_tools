#%%

import lunar_tools as lt
gpt4 = lt.GPT4()
msg = gpt4.generate("tell me about yourself")
# %%
msg
# %%

from utils import *

read_api_key('openai')
# %%


save_api_key_to_lunar_config('OPEN_AI_KEY','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')
# %%
