from datasets import load_dataset
from transformers import AutoTokenizer

# =========================================================
# LOAD DATASET
# =========================================================

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)

print(dataset)

# =========================================================
# REDUCE DATASET SIZE
# =========================================================

small_dataset = dataset["train"].select(range(5000))

print("Dataset Size:", len(small_dataset))

# =========================================================
# FORMAT DATASET
# =========================================================

def format_data(example):

    text = f"""
### Instruction:
{example['instruction']}

### Response:
{example['response']}
"""

    return {
        "text": text
    }

formatted_dataset = small_dataset.map(format_data)

print(formatted_dataset[0]["text"])

# =========================================================
# LOAD TOKENIZER
# =========================================================

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

# =========================================================
# TOKENIZATION
# =========================================================

def tokenize_function(example):

    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenized_dataset = formatted_dataset.map(
    tokenize_function
)

print("Tokenization Completed")

# =========================================================
# FINAL DATASET
# =========================================================

print(tokenized_dataset)
