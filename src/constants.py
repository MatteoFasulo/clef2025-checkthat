from transformers import pipeline

SEED = 42
EPOCHS = 6
BATCH_SIZE = 16
LR = 1e-5

DATA_PATH = "data/"
RESULTS_PATH = "results/"

SENT_PIPE = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    top_k=None,
)
