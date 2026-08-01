import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# 1. Load Pre-trained GPT-2 Model & Tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# 2. Sample Training Data
sample_text_data = [
    "Artificial intelligence is transforming how we interact with technology and solve complex problems.",
    "Machine learning models learn patterns directly from data to make accurate predictions.",
    "Fine-tuning large language models allows them to adapt to specific domains and writing styles.",
    "Generative AI models can produce coherent prose, poetry, code, and creative essays.",
    "Natural language processing bridges the gap between human communication and computational understanding."
]

# Convert text list to Hugging Face Dataset
dataset = Dataset.from_dict({"text": sample_text_data})

# Tokenize Data
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    logging_steps=5,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available()
)

# Data Collator for Causal Language Modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# 4. Initialize Trainer & Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

print("\n--- Starting Fine-Tuning ---")
trainer.train()

# 5. Text Generation Function
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test Text Generation
test_prompt = "Artificial intelligence"
print("\n--- Generated Output ---")
print(generate_text(test_prompt))