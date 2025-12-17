# BERT Phishing Email Detection

A deep learning–based phishing email detection system built using **BERT (Transformer architecture)** to accurately classify emails as **phishing** or **legitimate** by understanding contextual language patterns.

---

##  Live Model (Hugging Face)

The trained model is hosted on Hugging Face Hub and can be used directly for inference:

👉 **Hugging Face Model:**  
https://huggingface.co/divyanshu-chauhan-7786/bert-phishing-email-detection

The model also supports the **Hugging Face Inference API** for real-time predictions without managing infrastructure.

---

##  Project Overview

Phishing emails are a major cybersecurity threat that traditional rule-based systems often fail to detect. This project leverages **Transformer-based NLP** to identify phishing attempts using contextual understanding rather than keyword matching.

### Key Features
- Transformer-based **BERT** architecture
- Context-aware phishing detection
- Length-based outlier handling (256 tokens)
- High precision and recall
- Production-ready model deployment on Hugging Face

---

##  Model Performance

The model was evaluated using multiple strategies:

- **Accuracy:** ~99%
- **Precision:** ~99.5%
- **Recall:** ~99.3%
- **F1-score:** ~99.4%

High recall ensures minimal false negatives, which is critical for phishing detection systems.

---

##  Example Usage

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained(
    "divyanshu-chauhan-7786/bert-phishing-email-detection"
)
tokenizer = BertTokenizer.from_pretrained(
    "divyanshu-chauhan-7786/bert-phishing-email-detection"
)

def predict_email(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    return "Phishing" if logits.argmax().item() == 1 else "Legitimate"

predict_email("Urgent! Verify your bank account immediately.")
```

---
##  Tech Stack

- Python  
- PyTorch  
- Hugging Face Transformers  
- Hugging Face Datasets  
- Scikit-learn  
- Matplotlib  

---

##  Limitations

- Emails longer than 256 tokens are truncated
- Only English-language emails are supported
- The model processes email body text only (headers and metadata are not included)

---

##  Future Improvements

- Support for long-context Transformer models such as **Longformer** and **BigBird**
- Incorporation of email headers and metadata for improved detection accuracy
- Extension to multilingual phishing email detection
- Deployment as a web application or API-based service

---

##  Author

**Divyanshu Chauhan**  
AI Engineer | Data Analyst | Machine Learning & Deep Learning  

🔗 **Portfolio:**  
https://divyanshu-chauhan-7786.github.io/divyanshu-chauhan/
