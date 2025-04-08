import torch
from transformers import AutoModel
from huggingface_hub import PyTorchModelHubMixin

class MultilabelModel(torch.nn.Module, PyTorchModelHubMixin):
    
    def __init__(self, model_name, config):
        super(MultilabelModel, self).__init__()
        self.model_name = model_name
        self.config = config
        self.n_classes = config['n_classes']
        self.dropout_rate = config['dropout_rate']
        self.bert = AutoModel.from_pretrained(self.model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.classifier = torch.nn.Linear(768, self.n_classes)
    
    def forward(self, ids, mask, token_type_ids):
        output = self.bert(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = output[0]
        output = output[:, 0]
        output = self.pre_classifier(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.classifier(output)
        return output



