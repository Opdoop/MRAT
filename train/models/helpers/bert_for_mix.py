"""
LSTM 4 Classification
^^^^^^^^^^^^^^^^^^^^^^^

"""
import torch.nn as nn
import transformers
from train.models.helpers.utils import load_cached_state_dict
from train.shared import utils

class MixBert(nn.Module):
    def __init__(self, model, num_labels, finetuning_task):
        super(MixBert, self).__init__()

        self.config = transformers.AutoConfig.from_pretrained(
            model, num_labels=num_labels, finetuning_task=finetuning_task, output_hidden_states=True
        )
        self.bert = transformers.AutoModelForSequenceClassification.from_pretrained(
            model,
            config=self.config,
        )

        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))

    def forward(self, _input, perturbed_input=None, mix_ratio=None):

        if perturbed_input == None:
            output = self.bert(**_input)
            last_hidden_state = output.hidden_states[12][:,0,:]  # last layer [cls]
            pred = self.linear(last_hidden_state)
        else:
            output_normal = self.bert(**_input).hidden_states[12][:,0,:]
            output_perturb = self.bert(**perturbed_input).hidden_states[12][:,0,:]
            output = (1 - mix_ratio) * output_perturb + mix_ratio * output_normal
            pred = self.linear(output)
        return pred

    def load_from_disk(self, model_path):
        self.load_state_dict(load_cached_state_dict(model_path))
        self.to(utils.device)
        self.eval()