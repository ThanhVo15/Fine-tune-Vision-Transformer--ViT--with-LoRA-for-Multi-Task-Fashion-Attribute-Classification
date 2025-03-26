**Fine-tune-Vision-Transformer--ViT--with-LoRA-for-Multi-Task-Fashion-Attribute-Classification**  
A practical example of leveraging [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) to fine-tune a Vision Transformer (ViT) for multi-task fashion attribute classification. This repository covers data preparation, feature engineering, model building, and fine-tuning for classifying various attributes (gender, category hierarchy, color, brand, etc.) of a fashion product image.

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Project Overview](#project-overview)  
3. [Requirements & Installation](#requirements--installation)  
4. [Dataset](#dataset)  
5. [Data Preprocessing](#data-preprocessing)  
6. [Model Architecture](#model-architecture)  
7. [LoRA Integration](#lora-integration)  
8. [Training & Evaluation](#training--evaluation)  
9. [Usage](#usage)  
10. [Results](#results)  
11. [References](#references)  
12. [License](#license)

---

## Introduction
Large-scale Vision Transformers (ViT) often require significant GPU memory and computation for fine-tuning. [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) addresses this by injecting low-rank updates into the model’s attention weights, drastically reducing the number of trainable parameters. This project demonstrates how to:
1. **Load and preprocess** the Fashion Product Images dataset.
2. **Incorporate multi-task outputs** (gender, masterCategory, subCategory, usage, baseColour, brand, etc.).
3. **Inject LoRA modules** into a frozen ViT backbone to enable efficient fine-tuning.
4. **Train & evaluate** the multi-task model on new, custom attributes.

---

## Project Overview
The goal is to classify various attributes of a fashion product image in a hierarchical manner:
- **Level 1**: Predict `gender`, `masterCategory`, and `usage`.
- **Level 2**: Predict `subCategory` given the knowledge of level 1 predictions.
- **Level 3**: Predict `articleType` given previous levels.
- **Level 4**: Predict `baseColour` and `brand`.

During inference, a single input image is processed by the Vision Transformer, which outputs predictions for all these attributes. By employing LoRA, we only update a small fraction of the parameters (the low-rank decomposition layers) while keeping the rest of the ViT model frozen.

---

## Requirements & Installation
- Python 3.7+
- [PyTorch](https://pytorch.org/) (with CUDA if using a GPU)
- [Transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets)
- [evaluate](https://github.com/huggingface/evaluate)
- [torchvision](https://pytorch.org/vision/stable/)
- [accelerate](https://github.com/huggingface/accelerate)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) (if quantization or 8-bit is used)
- [peft](https://github.com/huggingface/peft) (LoRA integration)

Install required packages:
```bash
pip install datasets torchvision accelerate evaluate peft bitsandbytes transformers
```

---

## Dataset
We use the [Fashion Product Images Dataset](https://huggingface.co/datasets/ashraq/fashion-product-images-small). It contains product images and metadata such as:
- `gender`
- `masterCategory`
- `subCategory`
- `articleType`
- `baseColour`
- `season`
- `year`
- `usage`
- `productDisplayName`

Each image and metadata entry describes a single fashion product. We add a custom `price_in_vnd` field (simulated), split the dataset into train/validation/test, and apply label encoding or numeric scaling where appropriate.

---

## Data Preprocessing
1. **Label Encoding**: Convert categorical columns (`gender`, `masterCategory`, `subCategory`, etc.) into numeric labels.
2. **Text Tokenization**: Use a BERT tokenizer on `productDisplayName` if text inputs are involved in the multi-task scenario.
3. **Image Processing**: 
   - Resize to `(224, 224)`.
   - Normalize with the mean and std from a standard ViT `feature_extractor`.
4. **Data Collation**: A custom collator function (`data_collator`) collects images and tokenized text, returning them in batch-friendly tensors.

---

## Model Architecture
We start with a [ViTModel](https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py) (`google/vit-base-patch16-224` as default).  
**Key Steps**:
- Freeze the base ViT parameters (no gradient updates).
- Add hierarchical classification heads:
  - **Level 1** outputs: gender, masterCategory, usage.
  - **Level 2** uses outputs from Level 1 + ViT embedding to predict subCategory.
  - **Level 3** uses outputs from previous levels + ViT embedding to predict articleType.
  - **Level 4** outputs baseColour and brand.

```python
class CustomViTHierarchical(nn.Module):
    def __init__(...):
        self.vit = vit_model
        # Additional classification heads:
        self.head1_gender = ...
        self.head1_master = ...
        self.head1_usage  = ...
        ...
```

---

## LoRA Integration
Using [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft), we integrate LoRA into the ViT attention layers:
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query","key","value","dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="FEATURE_EXTRACTION"
)

model = get_peft_model(model, lora_config)
```
This drastically reduces the trainable parameters and GPU memory usage, making fine-tuning more accessible.

---

## Training & Evaluation
We use Hugging Face’s [Trainer API](https://github.com/huggingface/transformers/tree/main/src/transformers/trainer.py) with a custom `CustomTrainer` and custom `compute_loss`.  
**Key training arguments**:
```python
training_args = TrainingArguments(
    output_dir="./vit_hierarchical",
    ...
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=True,
    num_train_epochs=5,
    ...
)
```
**Steps**:
1. Instantiate the Trainer with your model, datasets, data collator, and `compute_metrics`.
2. Call `trainer.train()` to begin fine-tuning.
3. Evaluate on the validation/test sets.

---

## Usage
Once fine-tuned, you can load the model and pass an image through the pipeline. Example:
```python
from transformers import ViTModel, BertTokenizer
from PIL import Image

model = ViTModel.from_pretrained("./my_finetuned_vit")
tokenizer = BertTokenizer.from_pretrained("./my_finetuned_vit")
image = Image.open("sample_fashion_item.jpg")

# Perform transformations: resize, normalize, to tensor...
# Then call model or your custom classification module
# to get multi-attribute predictions.
```
This will produce predictions for gender, category, usage, subCategory, articleType, baseColour, and brand.

---

## Results
- **Multi-Task Accuracy**: Observed stable improvements in each attribute classification compared to a single-task approach.
- **Parameter Efficiency**: With LoRA, only a small fraction of parameters are updated, reducing GPU memory usage significantly.
- **Scalability**: This approach can easily be extended to more attributes or tasks.

---

## References
- [Understanding Low-Rank Adaptation (LoRA) for Efficient Fine-tuning of Large Language Models](https://ai.plainenglish.io/understanding-low-rank-adaptation-lora-for-efficient-fine-tuning-of-large-language-models-082d223bb6db)  
- [Practical Guide to Fine-Tune LLMs with LoRA](https://medium.com/@manindersingh120996/practical-guide-to-fine-tune-llms-with-lora-c835a99d7593)  
- [Fine-tuning a Vision Transformer (ViT) Model with a Custom Dataset](https://medium.com/@imabhi1216/fine-tuning-a-vision-transformer-vit-model-with-a-custom-dataset-37840e4e9268)

---

## License
This repository is released under the [MIT License](./LICENSE).  
Feel free to modify and adapt to your use case.