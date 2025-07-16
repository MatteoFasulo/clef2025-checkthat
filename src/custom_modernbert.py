import torch
import torch.nn as nn
from transformers import ModernBertPreTrainedModel, ModernBertModel, ModernBertConfig
from transformers.activations import GELUActivation


class CustomModernBertModel(ModernBertPreTrainedModel):
    config_class = ModernBertConfig

    def __init__(self, config, sentiment_dim=3, num_labels=2, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.config = config
        self.model = ModernBertModel(config)
        self.head = torch.nn.Sequential(
            nn.Linear(config.hidden_size + sentiment_dim, config.hidden_size + sentiment_dim, config.classifier_bias),
            GELUActivation(),
            nn.LayerNorm(config.hidden_size + sentiment_dim, eps=config.norm_eps, bias=config.norm_bias),
        )

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(config.hidden_size + sentiment_dim, num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids, positive, neutral, negative, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0]

        if self.config.classifier_pooling == "cls":
            last_hidden_state = last_hidden_state[:, 0]
        elif self.config.classifier_pooling == "mean":
            last_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                dim=1, keepdim=True
            )

        # Sentiment features as a single tensor
        sentiment_features = torch.stack((positive, neutral, negative), dim=1)  # Shape: (batch_size, 3)

        # Combine output embedding with sentiment features
        combined_features = torch.cat((last_hidden_state, sentiment_features), dim=1)

        pooled_output = self.head(combined_features)

        # Classification head
        logits = self.classifier(self.dropout(pooled_output))

        return {"logits": logits}
