from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM

# Load dataset
dataset = load_dataset("csv", data_files="data/historical_emails.csv")
dataset = dataset.train_test_split(test_size=0.1)

# Load Mistral Model
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# Define Training Parameters
training_args = TrainingArguments(
    output_dir="./models/mistral_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Train Model
trainer.train()

# Save Fine-tuned Model
model.save_pretrained("models/mistral_finetuned")
