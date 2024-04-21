from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm
from .. import clf_path

import pickle
import sys


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

model = ControlModel(model, list(range(-5, -18, -1)))
user_tag, asst_tag = "[INST]", "[/INST]"



#clf, vec = pickle.load(open(clf_path, 'rb'))
#print('read clf %s' % str(clf))
#print('read vec %s' % str(vec



@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
	form = MyForm()
	result = None
	if form.validate_on_submit():
		vector_path = form.text_options.data

		control_vector = torch.load(vector_path)

		input_field = form.input_field.data

		input_query = user_tag + input_field + asst_tag

		print("magnitude: %s" % form.magnitude.data)

		mag = float(form.magnitude.data)

		input_ids = tokenizer(input_query, return_tensors="pt").to(model.device)
		settings = {
			"pad_token_id": tokenizer.eos_token_id, # silence warning
			"do_sample": False, # temperature=0
			"max_new_tokens": 512,
			"repetition_penalty": 1.1, # reduce control jank
		}

		print("==baseline")
		model.reset()
		default_output = str(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()))


		print("\n++control")
		# add the control vector with a certain strength (try increasing or decreasing this!)
		model.set_control(control_vector, mag)
		control_output = str(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()))

		model.reset()
		#X = vec.transform([input_field])
		#pred = clf.predict(X)[0]
		pred = "PRED"
		#proba = clf.predict_proba(X)[0].max()
		proba = 0.5
		# flash(input_field)
		return render_template('myform.html', title='', form=form, 
								prediction=control_output, confidence='%.2f' % proba)
		#return redirect('/index')
	return render_template('myform.html', title='', form=form, prediction=None, confidence=None)
