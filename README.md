# 📊 SmolLM3 Visualization Code Generator (LoRA SFT)

Fine-tuning **SmolLM3-3B** using Supervised Fine-Tuning (SFT) + LoRA (PEFT) to generate high-quality Python data visualization and analysis code.

---

## 🚀 Model Overview

- **Base Model:** HuggingFaceTB/SmolLM3-3B  
- **Training Method:** Supervised Fine-Tuning (SFT)  
- **PEFT Method:** LoRA (Low-Rank Adaptation)  
- **Dataset:** sahil2801/CodeAlpaca-20k (filtered)  
- **Specialization:** Python Data Visualization & Data Analysis Code  

---

## 🎯 Project Goal

This project specializes SmolLM3-3B into a focused **Data Visualization Code Generator** that:

- Generates clean Python code
- Uses:
  - `matplotlib`
  - `seaborn`
  - `pandas`
  - `numpy`
- Handles:
  - Line plots
  - Bar charts
  - Histograms
  - Scatter plots
  - Boxplots
  - Heatmaps
  - Statistical summaries
  - CSV/DataFrame analysis

---

# 🧠 Training Pipeline

## 1️⃣ Dataset Loading

```python
from datasets import load_dataset

data = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
```

---

## 2️⃣ Dataset Filtering

Filtered samples related to visualization & analysis using keywords like:

- matplotlib
- seaborn
- plot
- chart
- histogram
- dataframe
- statistics
- mean, median, correlation

---

## 3️⃣ Prompt Formatting

Each example is converted into structured instruction format:

```
### Instruction:
...

### Input:
...

### Response:
...
```

---

## 4️⃣ Model Setup

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM3-3B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

---

## 5️⃣ LoRA Configuration

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

### Why LoRA?

- Efficient fine-tuning
- Lower GPU memory usage
- Faster training
- Smaller upload size

---

## 6️⃣ Training Configuration

```python
from trl import SFTConfig

config = SFTConfig(
    output_dir="./smollm3-visualization-sft",
    per_device_train_batch_size=4,
    learning_rate=5e-4,
    num_train_epochs=2,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    dataset_text_field="text"
)
```

Trainer used: `SFTTrainer` from `trl`

---

# 💾 Saving & Uploading

## Save Locally

- Full model → `./smollm3-visualization-sft/final`
- LoRA adapter → `lora_adapter/`

## Upload to Hugging Face Hub

```python
trainer.push_to_hub("Kyrollos32/smollm3-CodeGenerator")
tokenizer.push_to_hub("Kyrollos32/smollm3-CodeGenerator")
```

---

# 🔥 Inference Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM3-3B",
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "Kyrollos32/smollm_sft_2"
)

model.eval()

prompt = """
Write Python code using matplotlib to create a line plot
for a DataFrame column named 'sales'.
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2,
        top_p=0.9
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

# 📦 Dependencies

```bash
pip install transformers datasets trl peft accelerate huggingface_hub torch
```

---

# 🏗 Project Structure

```
.
├── training_script.py
├── lora_adapter/
├── smollm3-visualization-sft/
│   └── final/
└── README.md
```

---

# 📈 Expected Capabilities

- Generate matplotlib plots  
- Work with pandas DataFrames  
- Perform statistical analysis  
- Create clean, runnable Python visualization scripts  

---

# 📜 License

Follow the license of:
- SmolLM3-3B
- CodeAlpaca-20k dataset

---

# 👨‍💻 Author

Fine-tuned & published by **Kyrollos32**