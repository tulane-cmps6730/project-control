import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append(os.path.join(os.getcwd(), "../libraries/repeng"))
from repeng import ControlVector, ControlModel, DatasetEntry


def query_model(query, control_vector, magnitude):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    user_tag, asst_tag = "[INST]", "[/INST]"
    # the question to ask the modified model
    # don't forget the space after {user_tag} and before {asst_tag}!
    # input = f"{user_tag} What are human beings like? {asst_tag}"
    input = user_tag + " " + query + " " + asst_tag
    
    # tokenizer and generation settings
    input_ids = tokenizer(input, return_tensors="pt").to(model.device)
    settings = {
        "pad_token_id": tokenizer.eos_token_id, # silence warning
        "do_sample": False, # temperature=0
        "max_new_tokens": 128,
        "repetition_penalty": 1.1, # reduce control jank
    }
    
    # add the control vector with a certain strength (try increasing or decreasing this!)
    model.set_control(control_vector, magnitude)
    response = tokenizer.decode(model.generate(**input_ids, **settings).squeeze())
    return response