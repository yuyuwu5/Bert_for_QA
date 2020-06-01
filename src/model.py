import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel

class BertForQA(BertPreTrainedModel):
	def __init__(self, config, num_labels=1):
		super(BertForQA, self).__init__(config)
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.linear = nn.Linear(config.hidden_size, num_labels)
		self.qa_start = nn.Linear(config.hidden_size, 1)
		self.qa_end = nn.Linear(config.hidden_size, 1)
		self.init_weights()

	def forward(self, contextQuestion, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
		output = self.bert(contextQuestion, attention_mask=attention_mask, token_type_ids=token_type_ids)
		seq_output = output[0]
		pool_output = output[1]
		answerable = self.linear(self.dropout(pool_output))
		start = self.qa_start(seq_output)
		end = self.qa_end(seq_output)
		return answerable, start, end
