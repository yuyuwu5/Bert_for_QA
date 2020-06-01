import os
import json
import torch
import pickle
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import BertForQA
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer 
from torch import optim
from torch.utils.data import Dataset, DataLoader
from buildDataset import QADataset 

TRAIN_DATA = "../data/train.pkl"
DEV_DATA = "../data/dev.pkl"
PRETRAINED_MODEL_NAME = "bert-base-chinese"
PREDICTION = "./result/"
SAVE_MODEL_DIR = "../model/"
NUM_LABELS = 1
BATCH_SIZE = 6
VAL_BATCH_SIZE = 16
EPOCH = 5
LEARNING_RATE = 2e-5
POS_WEIGHT = 0.48 #0.42857142857142855 
LEN = 512
SEARCH = 5

#torch.cuda.set_device(2)

def calculate_loss(answerable_predict, answerable, start_predict, start, end_predict, end, token_type, attention_mask, device, loss_answerable, loss_start, loss_end):
	ans_loss = loss_answerable(answerable_predict, answerable)
	mask = attention_mask^token_type
	mask = F.pad(mask, (0, LEN-attention_mask.shape[1]), "constant", 0)
	inf = torch.tensor([[float("-Inf")]*LEN]*answerable_predict.shape[0]).to(device)
	start_predict = F.pad(start_predict, (0, LEN-attention_mask.shape[1]), "constant", 0)
	start_predict = torch.where(mask==1, start_predict, inf)
	end_predict = F.pad(end_predict, (0, LEN-attention_mask.shape[1]), "constant", 0)
	end_predict = torch.where(mask==1, end_predict, inf)
	start_loss = loss_start(start_predict, start) 
	end_loss = loss_end(end_predict, end)
	return ans_loss + start_loss + end_loss

def main(): 
	torch.set_printoptions(threshold=1000000)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logging.info("Device type is '%s'" %(device))
	logging.info("Load training data")
	with open(TRAIN_DATA, "rb") as f:
		trainData = pickle.load(f)
	logging.info("Build training generator")
	training_generator = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True, collate_fn=trainData.collate_fn)
	logging.info("Load dev data")
	with open(DEV_DATA, "rb") as f:
		devData = pickle.load(f)
	logging.info("Build dev generator")
	dev_generator = DataLoader(devData, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=devData.collate_fn)
	logging.info("Build Bert Model")
	model = BertForQA.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
	model.to(device)
	optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
	total_step = len(training_generator) * EPOCH
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps=total_step)
	loss_answerable = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT).to(device))
	loss_start = nn.CrossEntropyLoss(ignore_index=-1)
	loss_end = nn.CrossEntropyLoss(ignore_index=-1)
	#performance = 1000000000
	tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, do_lower_case=True)
	for c in range(EPOCH):
		epoch_loss = 0
		for step, data in enumerate(training_generator):
			model.train()
			optimizer.zero_grad()
			idx = data['id']
			content = data['content_question'].to(device, dtype=torch.long)
			attention_mask = data['attention_mask'].to(device, dtype=torch.long)
			token_type = data['token_type_id'].to(device, dtype=torch.long)
			answerable = data['answerable'].to(device, dtype=torch.float)
			start = data['start'].to(device, dtype=torch.long)
			end = data['end'].to(device, dtype=torch.long)
			answerable_predict, start_predict, end_predict = model(content, attention_mask=attention_mask, token_type_ids=token_type)
			answerable_predict = answerable_predict.squeeze(1)
			start_predict = start_predict.squeeze(2)
			end_predict = end_predict.squeeze(2)

			loss = calculate_loss(answerable_predict, answerable, start_predict, start, end_predict, end, token_type, attention_mask, device, loss_answerable, loss_start, loss_end)

			epoch_loss += loss.item()
			loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
			optimizer.step()
			scheduler.step()
			if step%10 == 0:
				logging.info("Epoch %s, step %s, Ein=%s"%(c, step, epoch_loss/(step+1)))
		model.eval()
		ans = {}
		val_loss = 0
		logging.info("Start validation")
		for step, data in enumerate(dev_generator):
			if step % 20 == 0:
				logging.info("Valid step %s" %(step))
			with torch.no_grad():
				idx = data['id']
				content = data['content_question'].to(device, dtype=torch.long)
				attention_mask = data['attention_mask'].to(device, dtype=torch.long)
				token_type = data['token_type_id'].to(device, dtype=torch.long)
				answerable_predict, start_predict, end_predict = model(content, attention_mask=attention_mask, token_type_ids=token_type)
				answerable_predict = answerable_predict.squeeze(1)
				start_predict = start_predict.squeeze(2)
				end_predict = end_predict.squeeze(2)
				answerable = data['answerable'].to(device, dtype=torch.float)
				start = data['start'].to(device, dtype=torch.long)
				end = data['end'].to(device, dtype=torch.long)
				loss = calculate_loss(answerable_predict, answerable, start_predict, start, end_predict, end, token_type, attention_mask, device, loss_answerable, loss_start, loss_end)
				val_loss += loss.item()
				answerable_predict = torch.sigmoid(answerable_predict)
				mask = attention_mask^token_type
				mask = F.pad(mask, (0, LEN-attention_mask.shape[1]), "constant", 0)
				inf = torch.tensor([[float("-Inf")]*LEN]*answerable_predict.shape[0]).to(device)
				start_predict = F.pad(start_predict, (0, LEN-attention_mask.shape[1]), "constant", 0)
				start_predict = torch.where(mask==1, start_predict, inf)
				end_predict = F.pad(end_predict, (0, LEN-attention_mask.shape[1]), "constant", 0)
				end_predict = torch.where(mask==1, end_predict, inf)
				l = len(answerable_predict)
				st_v, st_idx = torch.topk(start_predict, SEARCH)
				ed_v, ed_idx = torch.topk(end_predict, SEARCH)
				#print(st_idx.shape)
				out_range = attention_mask ^ token_type
				out_range = out_range.sum(1)
				for i in range(l):
					s = 0
					e = 0
					while (s < SEARCH-1 and e < SEARCH-1) and ((st_idx[i,s] >= out_range[i]) or (st_idx[i,s] >= ed_idx[i, e]) or (ed_idx[i, e]-st_idx[i,s] > 30)):
						if st_v[i,s] >= ed_v[i,e]: 
							e += 1
						else:
							s += 1
					if ed_idx[i,e] >= out_range[i]:
						ed_idx[i,e] = out_range[i].item()-1
					if answerable_predict[i] < 0.5 or st_idx[i,s] >= out_range[i] or st_idx[i,s]>=ed_idx[i,e] or ed_idx[i, e]-st_idx[i,s]>30:
						ans[idx[i]] = ""
					else:
						ans[idx[i]] = "".join(tokenizer.convert_ids_to_tokens(content[i][st_idx[i,s]:ed_idx[i, e]]))
						ans[idx[i]] = ans[idx[i]].replace("#","")
						ans[idx[i]] = ans[idx[i]].replace("[UNK]","")
		logging.info("End validation with overall Eval %s" %(val_loss / len(dev_generator)))
		#if val_loss < performance:
		#	performance = val_loss
		ckpt_path = "%sckpt%s.pt"%(SAVE_MODEL_DIR, c)
		logging.info("Save model to %s" %(ckpt_path))
		torch.save({
			"state_dict": model.state_dict(),
			"epoch":c
			}, ckpt_path
		)
		with open("%s%s_0.9.json"%(PREDICTION,c), "w") as f:
			json.dump(ans, f)

	return

if __name__=="__main__":
	loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
	logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
	main()
