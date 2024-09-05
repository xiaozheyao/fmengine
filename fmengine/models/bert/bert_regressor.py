from transformers import BertPreTrainedModel, BertModel


class BertRegresser(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
