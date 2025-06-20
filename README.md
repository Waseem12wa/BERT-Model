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
