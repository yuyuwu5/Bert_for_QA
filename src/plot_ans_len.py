import matplotlib.pyplot as plt
import numpy as np
import json
from transformers import BertTokenizer

TRAIN = "../data/train.json"
MODEL = "bert-base-chinese"

def main():
	length = []
	with open(TRAIN) as f:
		train = json.load(f)
	tokenizer = BertTokenizer.from_pretrained(MODEL, do_lower_case=True)
	for data in train['data']:
		for sep in data['paragraphs']:
			for q in sep['qas']:
				if q['answerable']:
					length.append(len(tokenizer.tokenize(q['answers'][0]['text'])))
	plt.hist(length, 60, cumulative=True, density=True)
	plt.title("Cumulative Answer Length")
	plt.xlabel("Length")
	plt.ylabel("Count(%)")
	plt.savefig("culmulative_ans.png")
	return

if __name__ == "__main__":
	main()
