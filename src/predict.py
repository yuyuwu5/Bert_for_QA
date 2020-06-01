import os
import json
import torch
import logging
import buildDatasetForTest
import torch.nn.functional as F
from model import BertForQA
from transformers import BertTokenizer 
from torch.utils.data import Dataset, DataLoader
from buildDatasetForTest import QADataset 
from argparse import ArgumentParser

PRETRAINED_MODEL_NAME = "bert-base-chinese"
MODEL_PATH = "./model/best.pt"
NUM_LABELS = 1
VAL_BATCH_SIZE = 64
SEARCH = 5
LEN = 512

def main(args): 
	torch.set_printoptions(threshold=1000000)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logging.info("Device type is '%s'" %(device))
	logging.info("Load dev data")
	devData = buildDatasetForTest.buildQADataset(args.test_data_path)
	logging.info("Build dev generator")
	dev_generator = DataLoader(devData, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=devData.collate_fn)
	logging.info("Build Bert Model")
	model = BertForQA.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
	model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
	model.to(device)
	tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, do_lower_case=True)
	model.eval()
	ans = {}
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
	with open(args.output_path, "w") as f:
		json.dump(ans, f)

	return

def parse_argument():
	parser = ArgumentParser()
	parser.add_argument("--test_data_path", type=str)
	parser.add_argument("--output_path", type=str)
	args = parser.parse_args()
	return args

if __name__=="__main__":
	loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
	logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
	args = parse_argument()
	main(args)
