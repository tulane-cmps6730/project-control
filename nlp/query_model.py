import json
import sys
import torch
import os
from tqdm import tqdm
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.join(os.getcwd(), "./libraries/repeng"))

from repeng import ControlVector, ControlModel, DatasetEntry

# specify model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# specify tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = 0

# load model, move to dievice
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print ("Loaded to %s", device)



model = ControlModel(model, list(range(-5, -18, -1)))

user_tag, asst_tag = "[INST]", "[/INST]"

# sample input
input_query = "How do you create a new library in Python?"

# load control vector from cache
control_vector = torch.load("./nlp/data/vectors/Elaborate_Simple.pt")

# add inst tokens
input_query = user_tag + input_query + asst_tag

# tokenizer and generation settings
input_ids = tokenizer(input_query, return_tensors="pt").to(model.device)
settings = {
    "pad_token_id": tokenizer.eos_token_id, # silence warning
    "do_sample": False, # temperature=0
    "max_new_tokens": 128,
    "repetition_penalty": 1.1, # reduce control jank
}

print("==baseline")
model.reset()
print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()))

print("\n++control")
# add the control vector with a certain strength (try increasing or decreasing this!)
model.set_control(control_vector, 1.5)
print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()))

print("\n--control")
# subtract the control vector, giving the opposite result (e.g. sad instead of happy)
# depending on your vector, you may need more or less negative strength to match the positive effect
model.set_control(control_vector, -2.0)
print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()))
model.reset()
