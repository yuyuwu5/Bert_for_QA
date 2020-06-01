import os
import logging
import pickle
import torch
import json
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset

PRETRAINED_MODEL_NAME = "bert-base-chinese"
TRAIN_DATA_PATH = "../data/train.json"
DEV_DATA_PATH = "../data/dev.json"
OUTPUT_DIR = "../data/"
#"../data/sample_prediction.json"
#[CLS], [SEP], [UNK], [PAD], [MASK]
#MAX_ANS = 4.8
#MASK = tokenizer.convert_tokens_to_ids("[MASK]")


class QADataset(Dataset):
	def __init__(self, data, special, max_len=512):
		self.data = data
		self.special = special
		self.max_len = max_len
	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		sample = self.data[idx]
		instance = {
				"id": sample["id"],
				"question": sample["question"],
				"context": sample["context"]
				}
		if "answerable" in sample:
			instance["answerable"] = sample["answerable"]
			instance["ans_start"] = sample["ans_start"]
			instance["ans_end"] = sample["ans_end"]
		return instance
	def collate_fn(self, sample):
		idx = []
		content_question = []
		answerable = []
		start = []
		end = []
		attention_mask = []
		token_type_id = []
		maxL_Batch = max([len(s['context']+s['question']) for s in sample])
		maxL_Batch = min(maxL_Batch, self.max_len)
		for s in sample:
			idx.append(s['id'])
			att = np.ones(maxL_Batch)
			token_type = np.zeros(maxL_Batch)
			availL = min(maxL_Batch - len(s['question']) - 3, len(s['context']))
			qa = [self.special[0]] + [s['context'][i] for i in range(availL)] + [self.special[1]] + s['question'] + [self.special[1]]
			padL = max(maxL_Batch-len(qa), 0) 
			if padL > 0:
				token_type[availL+2:-padL] = 1
				att[-padL:] = 0
			else:
				token_type[availL+2:] = 1
			token_type_id.append(token_type)
			qa += [self.special[2] for i in range(padL)]
			attention_mask.append(att)
			content_question.append(qa)
			if "answerable" in s:
				if s['ans_start'] < availL and s['ans_end'] < availL:
					start.append(s['ans_start']+1)
					end.append(s['ans_end']+1)
					answerable.append(s['answerable'])
				elif s['ans_start'] < availL:
					start.append(s['ans_start']+1)
					end.append(availL)
					answerable.append(s['answerable'])
				else:
					start.append(-1)
					end.append(-1)
					answerable.append(0)
		batch = {
				'id': idx,
				'content_question': torch.tensor(content_question),
				'attention_mask': torch.tensor(attention_mask),
				'token_type_id': torch.tensor(token_type_id),
				}
		if "answerable" in sample[0]:
			batch['start'] = torch.tensor(start)
			batch['end'] = torch.tensor(end)
			batch['answerable'] = torch.tensor(answerable)
		return batch

def tokenize(data):
	tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, do_lower_case=True)
	CLS = tokenizer.convert_tokens_to_ids("[CLS]")
	SEP = tokenizer.convert_tokens_to_ids("[SEP]")
	PAD = tokenizer.convert_tokens_to_ids("[PAD]")
	dic = []
	for datas in data['data']:
		for sep in datas['paragraphs']:
			context = sep['context']
			context = tokenizer.tokenize(context)
			c_idx = tokenizer.convert_tokens_to_ids(context[:-1])
			for q in sep['qas']:
				question = q['question'][:-1]
				question = tokenizer.tokenize(question)
				if len(question)>50:
					question = question[:30] + question[-20:]
				q_idx = tokenizer.convert_tokens_to_ids(question)
				qa_set = {
						"id": q['id'],
						"question": q_idx,
						"context": c_idx,
						}
				if "answers" in q:
					qa_set["answerable"] = q["answerable"]
					if q["answerable"] == True:
						preToken = len(tokenizer.tokenize(sep['context'][:q["answers"][0]["answer_start"]]))
						qa_set["ans_start"] = preToken
						qa_set["ans_end"] = preToken  + len(tokenizer.tokenize(q["answers"][0]["text"]))
					else:
						qa_set['ans_start'] = -1
						qa_set['ans_end'] = -1
				dic.append(qa_set)
	return dic, [CLS, SEP, PAD]

def buildQADataset(data, out_path):
	logging.info("Build %s dataset" %(out_path))
	dataset = QADataset(data[0], data[1])
	with open(out_path, "wb") as f:
		pickle.dump(dataset, f)

def main():
	logging.info("Open Training Data")
	with open(TRAIN_DATA_PATH) as f:
		train = json.load(f)
	trainData= tokenize(train)
	buildQADataset(trainData, OUTPUT_DIR+"train.pkl")
	logging.info("Open Dev Data")
	with open(DEV_DATA_PATH) as f:
		dev = json.load(f)
	devData= tokenize(dev)
	buildQADataset(devData, OUTPUT_DIR+"dev.pkl")


if __name__ == "__main__":
	loglevel = os.environ.get('LOGLEVEL', "INFO").upper()
	logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
	main()
