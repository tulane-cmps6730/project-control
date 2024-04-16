import json
import sys
import torch
import os
from tqdm import tqdm
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer


sys.path.append(os.path.join(os.getcwd(), "../libraries/repeng"))



from repeng import ControlVector, ControlModel, DatasetEntry



def template(persona: str, suffix: str) -> str:
    return f"{user_tag} Act as if you're extremely {persona}. {asst_tag} {suffix}"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device Dectected: {device}")


model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = 0

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
model = ControlModel(model, list(range(-5, -18, -1)))

user_tag, asst_tag = "[INST]", "[/INST]"

print(f"Model Loaded: {model_name}")

topics_path = os.path.abspath(os.path.join(os.getcwd(), './data/topics.csv'))


with open("./data/train_data/all_truncated_outputs.json") as f:
    suffixes = json.load(f)

# Load topics

with open(topics_path, 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    topics = list(reader)
    
    # wrap the loop with tqdm
    for (synonym, antonym) in tqdm(topics, desc="Training Vectors"):
        tqdm_desc = f"Training Vector --> {synonym} | {antonym}"
        tqdm_instance = tqdm(total=len(suffixes), desc=tqdm_desc, position=1, leave=False)

        dataset = []

        positive_personas = [synonym]
        negative_personas = [antonym]

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

        tqdm_instance.update(1)
        tqdm_instance.set_description(tqdm_desc)

        model.reset() # make sure you always reset the model before training a new vector
        control_vector = ControlVector.train(model, tokenizer, dataset)

        control_vector.export_pytorch(f"./data/vectors/{synonym}_{antonym}.pt")

print("Training Complete, Shutting Down...")



