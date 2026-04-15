
import unsloth

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
dataset = load_dataset(
    "csv",
    data_files={
        "train": "../data/wikilarge_train.csv",
        "validation": "../data/wikilarge_validation.csv",
        "test": "../data/wikilarge_test.csv",   
    }
)

train_ds = dataset["train"]
val_ds = dataset["validation"]
test_ds = dataset["test"]

# Reduce dataset
train_ds = train_ds.select(range(50000))
def format_example(example):
    return {
        "text": f"""Simplify the following sentence into very simple English.
Make it easy for a 10-year-old to understand.
Use short sentences and very simple words.
Do NOT add extra information.

Sentence: {example['Normal']}
Simple: {example['Simple']}"""
}

train_ds = train_ds.map(format_example)
val_ds = val_ds.map(format_example)
test_ds = test_ds.map(format_example)

train_ds = train_ds.remove_columns(['Normal', 'Simple'])
val_ds = val_ds.remove_columns(['Normal', 'Simple'])
test_ds = test_ds.remove_columns(['Normal', 'Simple'])
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=256,
    dtype=torch.float16,
    load_in_4bit=True,
)

model.config.use_cache = False
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,
)
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
test_ds = val_ds.map(tokenize, batched=True)
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=1,   
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=training_args,
    data_collator=data_collator,
)
trainer.train()
val_ds = val_ds.remove_columns(["text"])
test_ds = test_ds.remove_columns(["text"])
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch

model.eval()

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ---------- VALIDATION ----------
val_loader = DataLoader(val_ds, batch_size=8, collate_fn=data_collator)

val_loss = 0
val_steps = 0

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model(**batch)
        val_loss += outputs.loss.item()
        val_steps += 1

print("Validation Loss:", val_loss / val_steps)


# ---------- TEST ----------
test_loader = DataLoader(test_ds, batch_size=8, collate_fn=data_collator)

test_loss = 0
test_steps = 0

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model(**batch)
        test_loss += outputs.loss.item()
        test_steps += 1

print("Test Loss:", test_loss / test_steps)
model.save_pretrained("simplifier-4090")
tokenizer.save_pretrained("simplifier-4090")
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="simplifier-4090",
    max_seq_length=256,
    dtype=torch.float16,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)
def simplify(text):
    prompt = f"Sentence: {text}\nSimple:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=True,
    temperature=0.1,   # 🔥 lower
    top_p=0.8,         # 🔥 tighter

    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    early_stopping=True,
)

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 🔥 CLEAN OUTPUT (IMPORTANT)
    return output.split("Simple:")[-1].strip()
test_examples = [
    # "The patient exhibited cardiovascular complications.",
    # "The committee reached a unanimous decision.",
    # "The company experienced a significant decline in revenue.",
    # "The experiment yielded inconclusive results.",
    # "The government implemented fiscal policies to mitigate the economic downturn.",
    # "The company leveraged strategic partnerships to enhance operational efficiency and scalability.",
    # "Despite facing unprecedented geopolitical tensions, the nation sustained its economic growth through diversified trade agreements.",
    # "The pharmaceutical intervention exhibited limited efficacy due to pharmacokinetic variability among patients."

     # 🏥 Medical
    "The patient exhibited symptoms consistent with acute respiratory distress syndrome.",
    "The treatment protocol was adjusted due to adverse pharmacological interactions.",

    # 💻 Technology
    "The system encountered a critical failure due to insufficient memory allocation.",
    "Artificial intelligence models require substantial computational resources for training.",

    # 📊 Economics / Finance
    "The central bank implemented monetary tightening to curb inflationary pressures.",
    "The company diversified its investment portfolio to mitigate financial risk.",

    # ⚖️ Legal
    "The defendant was acquitted due to lack of substantial evidence.",
    "The contract was deemed null and void due to breach of terms.",

    # 🌍 Geography / Environment
    "Climate change has exacerbated the frequency of extreme weather events globally.",
    "Deforestation has significantly impacted biodiversity in tropical regions.",

    # 🎓 Academic / Research
    "The study revealed a statistically significant correlation between the variables.",
    "The hypothesis was rejected after empirical data failed to support the theoretical model.",

    # 🏢 Business
    "The organization restructured its operations to improve overall productivity.",
    "The startup secured funding from venture capitalists to scale its operations.",

    # 🧠 Psychology
    "Cognitive behavioral therapy is effective in treating anxiety disorders.",
    "The experiment examined the impact of stress on human decision-making processes.",

    # ⚙️ Engineering
    "The structural integrity of the bridge was compromised due to material fatigue.",
    "The algorithm optimized performance by reducing computational complexity.",

    # 🛰️ Advanced / Mixed
    "The satellite communication system experienced latency due to signal interference.",
    "The geopolitical conflict disrupted global supply chains and economic stability."
]

for t in test_examples:
    print("INPUT :", t)
    print("OUTPUT:", simplify(t))
    print("-" * 50)