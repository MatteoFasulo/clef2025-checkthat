import torch
import torch.nn as nn
from transformers import PreTrainedModel, DebertaV2Model, DebertaV2Config
from transformers.models.deberta.modeling_deberta import ContextPooler


class CustomModel(PreTrainedModel):
    config_class = DebertaV2Config

    def __init__(self, config, sentiment_dim=3, num_labels=2, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(output_dim + sentiment_dim, num_labels)

    def forward(self, input_ids, positive, neutral, negative, attention_mask=None, labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)

        # Sentiment features as a single tensor
        sentiment_features = torch.stack((positive, neutral, negative), dim=1)  # Shape: (batch_size, 3)

        # Combine CLS embedding with sentiment features
        combined_features = torch.cat((pooled_output, sentiment_features), dim=1)

        # Classification head
        logits = self.classifier(self.dropout(combined_features))

        return {"logits": logits}
