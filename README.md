# 📰 Fake News Detector

A web-based machine learning app that detects whether a news headline or article is **Real** or **Fake** using Natural Language Processing and Logistic Regression. Built with Python, Scikit-learn, and Streamlit.

---

## 🚀 Features

- 🔍 Predict whether news is Real or Fake
- 📊 Show prediction confidence (%)
- 🧾 Accepts headline and full article text
- 💾 Trained on a labeled Kaggle dataset
- ⚙️ Model powered by TF-IDF + Logistic Regression
- 🌐 Lightweight web interface via Streamlit

---

## 🧠 Tech Stack

- **Python 3.11**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Streamlit**
- **TF-IDF Vectorizer**
- Logistic Regression Classifier

---

## 📈 Dataset

Used the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle:
- `True.csv` → Real news articles
- `Fake.csv` → Fake news articles

---

## 🛠️ Setup Instructions

1. **Clone this repo**  
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   
3. **Run the app**  
   ```bash
   streamlit run app.py
   
Example: 

    Try entering a suspicious news headline and see if it gets flagged!

    Try these examples: 

    “NASA confirms discovery of Earth-like planet in habitable zone of nearby star.”
    “Aliens endorse presidential candidate during secret UN meeting.”
    “Cows in Gujarat spotted using smartphones to trade cryptocurrency.”
