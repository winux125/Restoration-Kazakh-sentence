import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchcrf import CRF


class BestCRFClassifier(nn.Module):
    def __init__(self,model_name,num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)    
        self.dropout = nn.Dropout(0.1)
        self.linear  = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.crf = CRF(num_labels,batch_first=True)
    def forward(self, input_ids,attention_mask=None,**kwargs):
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        logits = self.linear(self.dropout(hidden))
        return TokenClassifierOutput(logits=logits)

