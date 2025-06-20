# BERT Resume Classifier (Fine-Tuned)

This model is a fine-tuned version of `bert-base-uncased` on a custom resume dataset containing resume texts and their associated job categories. It is designed to classify a resume into categories such as **Data Science**, **Web Development**, **HR**, and others.

---

## 📌 Model Details

- **Base Model:** [bert-base-uncased](https://huggingface.co/bert-base-uncased)
- **Architecture:** BERT + Classification Head
- **Task:** Multi-class Text Classification
- **Dataset Used:** Kaggle Resume Dataset (2 columns: `resume`, `category`)
- **Training Framework:** HuggingFace Transformers + Trainer API
- **Language:** English

---

## 🧠 Use Case

Given a resume in plain text format, this model predicts the **most suitable job category**.

**Example:**

```python
resume = "Experienced web developer skilled in JavaScript, React, and Node.js..."


🚀 Training Details
Trained for 3 epochs on custom dataset (80/20 split)

Batch size: 8

Optimizer and scheduler: Default in HuggingFace Trainer

Framework: PyTorch + Transformers

📁 Files Included
pytorch_model.bin - Fine-tuned weights

config.json - Model configuration

tokenizer/ - Tokenizer files

training_args.bin - Trainer settings (optional)

🧾 Citation
If you use this model, please cite the original paper for BERT:

Devlin et al. (2018), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

🙋 Author
👨‍💻 Waseem Zahid

🔬 Final Year CS Student & Research Assistant (Resume Intelligence Project)

🔗 Contact: waseem.zahid[at]email.com | FAST NUCES
