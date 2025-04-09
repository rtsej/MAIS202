# Political Bias Classifier – MAIS 202 Winter 2025

This project presents a tool to help users characterize the underlying bias of news articles they read. It uses a machine-learning model trained on a dataset of over 3,400 labeled articles to automatically classify underlying political bias (left, leaning left, center, right, leaning right) in a given piece of media.

## 📌 Project Motivation

Being able to identify fact vs. opinion has become more difficult in the current media landscape, and underlying political bias, even in articles not obviously political in nature, can be difficult to identify. In this context, we wanted to offer tools to empower readers to more confidently use information drawn from the media. This project aims to provide such a tool, using natural language processing (NLP) and deep learning to help readers identify bias in news texts.

## 📁 Dataset

We used a Kaggle dataset containing:
- `Title` – Headline of the article
- `Text` – Full content
- `Bias` – Label (left, lean left, center, lean right, right)

Link: [Kaggle Political Bias Dataset](https://www.kaggle.com/datasets/mayobanexsantana/political-bias/data)

## 🧠 Methodology

- **Text Preprocessing**: Cleaning, tokenization, BERT embeddings
- **Embeddings**: Generated using `bert-base-uncased` via HuggingFace Transformers
- **Model**: Custom PyTorch neural network (`BERT_Arch`) trained on embeddings
- **Training Split**: 70/15/15 train/validation/test using `train_test_split`

## 📊 Evaluation Metrics

- Mean Squared Error (MSE): 2.67
- F1 Scores: Between 0.55 and 0.71 across classes
- Additional evaluation via confusion matrix, accuracy, and recall

## 🛠️ Tech Stack

| Task | Tools Used |
|------|-------------|
| Data processing | pandas, scikit-learn |
| NLP | HuggingFace Transformers (BERT) |
| Modeling | PyTorch |
| Evaluation | scikit-learn metrics |
| Embedding generation | BERT (bert-base-uncased) |

## 🚀 How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/political-bias-classifier.git
   cd political-bias-classifier

Install dependencies

 pip install -r requirements.txt


Run the model training or inference script

 python train_model.py

<img width="561" alt="Screenshot 2025-04-08 at 7 59 18 PM" src="https://github.com/user-attachments/assets/e89e559c-1290-4d9e-b45d-98059a9bd585" />


We generated this confusion matrix using sklearn.metrics after several epochs of training. This chart makes it clear that there are some issues with our dataset. Namely, upon reviewing the article bias sheet again, it became clear that the vast majority of entries (1838 out of 3371) being classified as left, with an additional 519 entries being classified as leaning left. The effect of this is displayed in the confusion matrix above, as the model has learned that its best strategy is to almost always assume a left label, with only minor allowance given to right labeled articles whose contents are exceedingly dissimilar. To attain better results, it might be preferable to either work with a dataset with a broader array of articles, or to better tailor our training data to accommodate for this discrepancy.
