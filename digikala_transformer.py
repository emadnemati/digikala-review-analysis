
from torch import nn
from torch.nn import functional as F

REC_LABELS = {"recommended": 0, "not_recommended": 1, "no_idea": 2}
BAND_LABELS = {"low": 0, "medium": 1, "high": 2}
IGNORE_INDEX = -100

class DigikalaTransformer(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.encoder.encoder.layer[-1].requires_grad_(True)

        hidden_size = encoder.config.hidden_size
        self.dropout = nn.Dropout(encoder.config.hidden_dropout_prob)
        self.recommendation_head = nn.Linear(hidden_size, len(REC_LABELS))
        self.rating_head = nn.Linear(hidden_size, len(BAND_LABELS))

    def forward(self, **inputs):
        cls = self.encoder(**inputs).last_hidden_state[:, 0]
        features = self.dropout(cls)
        return self.recommendation_head(features), self.rating_head(features)

def masked_cross_entropy(logits, targets, weights=None):
    valid = targets.ne(IGNORE_INDEX)
    losses = F.cross_entropy(
        logits, targets, weight=weights, ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    denominator = valid.sum()
    if weights is not None:
        denominator = (weights[targets.clamp_min(0)] * valid).sum()
    return losses.sum() / denominator.clamp_min(1e-12)
