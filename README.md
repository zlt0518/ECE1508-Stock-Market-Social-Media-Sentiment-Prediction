# 📈 Stock-Sentiment: Social Media Sentiment Analysis for Stock Market Prediction

This project investigates the relationship between social media sentiment and stock market trends. By combining a fine-tuned **FinancialBERT** model for sentiment analysis with a **Random Forest** classifier, we aim to predict the direction of **Tesla (TSLA)** stock price movements.

## 🧠 Overview

Social media platforms such as Twitter (X) provide real-time insights into public market sentiment. In this project, we:

- Classify tweet sentiment as **Bullish**, **Neutral**, or **Bearish** using a fine-tuned FinancialBERT model.
- Combine the sentiment results with historical TSLA stock price data.
- Use a Random Forest classifier to predict the next-day stock price direction.

---

## 📂 Project Structure
- fintune_FinBERT_model.ipynb # Fine-tuning FinancialBERT on labeled tweets 
- fixed_randomforest_prediction.ipynb # Predicting TSLA stock movement using sentiment & price data 

---

## 🧰 Technologies

- Python 3
- HuggingFace Transformers (FinancialBERT)
- Scikit-learn (RandomForestClassifier)
- Pandas, NumPy, Matplotlib
- Jupyter Notebook

---

## 📊 Dataset Summary

| Dataset | Description |
|--------|-------------|
| `zeroshot/Twitter-financial-news-sentiment` | ~12,000 labeled finance-related tweets (Bullish, Neutral, Bearish) |
| Tesla Tweets | ~37,000 tweets related to Tesla |
| Stock Prices | TSLA stock data from 2021.09.30 to 2022.09.30 |

---

## 🔄 Model Pipeline

### Part 1: Sentiment Analysis
- Preprocess tweets (remove URLs, hashtags, etc.)
- Fine-tune FinancialBERT
- Predict sentiment for Tesla tweets

### Part 2: Stock Movement Prediction
- Merge sentiment results with TSLA stock price
- Create 2-week sentiment and price feature window
- Train Random Forest classifier to predict next-day movement

---

## ✅ Results

| Task | Accuracy |
|------|----------|
| Sentiment Classification (FinancialBERT) | 83.66% |
| Stock Prediction (Random Forest)         | 65.00% |

- ROC AUC: 0.64
- F1 Score (Sentiment): 0.83
- Macro-average F1 Score (Stock Prediction): 0.64

---

## 🔮 Future Work

- Incorporate real-time tweet metadata (retweets, likes) for weighted sentiment
- Use X/Twitter API for live prediction
- Explore LSTM or hybrid models for time-series improvement

---

## 👨‍💻 Team Members

- **Litao (John) Zhou** – Preprocessing, Fine-tuning FinancialBERT, Random Forest implementation  
- **Yuhan Zeng** – Dataset collection, Model buildup, Finetuning FinBERT, Prediction model build up  
- **Zeying Zhou** – Dataset collection, Model fine-tuning

📧 Contact: `{litao.zhou, yuhan.zeng, zeying.zhou}@mail.utoronto.ca`

---

## 📚 References

- [FinancialBERT - HuggingFace](https://huggingface.co/ahmedrachid/FinancialBERT-Sentiment-Analysis)  
- [Twitter Financial News Dataset](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)  
- [Stock Dataset (Kaggle)](https://www.kaggle.com/code/equinxx/stock-prediction-gan-twitter-sentiment-analysis/input)

---

## 📜 License and disclaimer

This project is intended for academic and research purposes only.
Please follow the academic integrity at
the University of Toronto
