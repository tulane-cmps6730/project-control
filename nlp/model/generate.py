import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append(os.path.join(os.getcwd(), "../libraries/repeng"))
from repeng import ControlVector, ControlModel, DatasetEntry
import tqdm

def template(persona: str, suffix: str) -> str:
    return f"{user_tag} Act as if you're extremely {persona}. {asst_tag} {suffix}"

def generate_vector(positive_personas, negative_personas, model_name = "mistralai/Mistral-7B-Instruct-v0.1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loaded to %s" %(device))
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ControlModel(model, list(range(-5, -18, -1)))
    
    user_tag, asst_tag = "[INST]", "[/INST]"


    with open("../data/train_data/all_truncated_outputs.json") as f:
        suffixes = json.load(f)

    dataset = []
    for suffix in suffixes:
        tokens = tokenizer.tokenize(suffix)
        for i in range(1, len(tokens)):
            truncated = tokenizer.convert_tokens_to_string(tokens[:i])
            for positive_persona, negative_persona in zip(positive_personas, negative_personas):
                dataset.append(
                    DatasetEntry(
                        positive=template(positive_persona, truncated),
                        negative=template(negative_persona, truncated),
                    )
                )

    model.reset() # make sure you always reset the model before training a new vector
    control_vector = ControlVector.train(
        model,
        tokenizer,
        dataset,
    )
    control_vector.export_pytorch("data/vectors/%s.pth" %(positive_personas[0]))

    