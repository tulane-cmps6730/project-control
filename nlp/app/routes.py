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




#clf, vec = pickle.load(open(clf_path, 'rb'))
#print('read clf %s' % str(clf))
#print('read vec %s' % str(vec



@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
	form = MyForm()
	result = None
	if form.validate_on_submit():
		input_field = form.input_field.data
		#X = vec.transform([input_field])
		#pred = clf.predict(X)[0]
		pred = "PRED"
		#proba = clf.predict_proba(X)[0].max()
		proba = 0.5
		# flash(input_field)
		return render_template('myform.html', title='', form=form, 
								prediction=labels[pred], confidence='%.2f' % proba)
		#return redirect('/index')
	return render_template('myform.html', title='', form=form, prediction=None, confidence=None)
