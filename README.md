# 📰 Fake News Detection using Machine Learning

A Machine Learning project that classifies news articles as **Fake** or **Real** using **TF-IDF Vectorization** and **Logistic Regression**. This project demonstrates the complete machine learning workflow, from data preprocessing to model training, evaluation, and prediction.

---

## 📖 Overview

Fake news has become a major challenge in the digital age. This project uses Natural Language Processing (NLP) and Machine Learning techniques to automatically classify news articles as **Fake** or **Real**.

The project performs:

- Data Loading
- Data Preprocessing
- Feature Extraction using TF-IDF
- Model Training using Logistic Regression
- Model Evaluation
- Fake/Real News Prediction

---

## ✨ Features

- 📂 Load dataset from CSV
- 🧹 Data preprocessing
- 🔤 Combine title and article text
- 📊 TF-IDF Vectorization
- 🤖 Logistic Regression classifier
- 📈 Accuracy calculation
- 📋 Classification Report
- 📉 Confusion Matrix visualization
- 📝 Predict custom news articles

---

## 🛠️ Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📁 Project Structure

```text
FakeNewsDetection/
│
├── images/
│   └── confusion_matrix.png
│
├── fake_news_detection.py
├── news.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Machine Learning Workflow

```text
News Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Combine Title + Text
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Logistic Regression Model
      │
      ▼
Prediction
      │
      ▼
Model Evaluation
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/244G5A3314THAHAHSIN/FakeNewsDetection.git
```

Move into the project folder:

```bash
cd FakeNewsDetection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python fake_news_detection.py
```

---

## 📊 Dataset

The dataset contains three columns:

| Column | Description |
|---------|-------------|
| title | News headline |
| text | Full news article |
| label | 0 = Fake, 1 = Real |

---

## 📈 Model Used

**Algorithm**

- Logistic Regression

**Feature Extraction**

- TF-IDF Vectorizer

---

## 📊 Results

| Metric | Score |
|---------|------:|
| Accuracy | 100% |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 Score | 1.00 |

> **Note:** These results are based on the sample dataset used in this project. Performance may vary with larger or real-world datasets.

---

## 📉 Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

---

## 💻 Example Prediction

**Input**

```text
Government launches new education policy across the country.
```

**Output**

```text
✅ REAL NEWS
```

---

## 🔮 Future Improvements

- Use a larger real-world dataset
- Compare additional algorithms (Random Forest, SVM, Naive Bayes)
- Build a Streamlit web application
- Deploy the model online
- Improve text preprocessing techniques

---

## 📚 Learning Outcomes

Through this project, I learned:

- Data preprocessing using Pandas
- Text feature extraction with TF-IDF
- Training a Logistic Regression model
- Evaluating classification models
- Visualizing results with a confusion matrix
- Managing code using Git and GitHub

---

## 👩‍💻 Author

**Syeda Thahasin Quasar**

GitHub: https://github.com/244G5A3314THAHAHSIN

---

## 📄 License

This project is created for educational and learning purposes.
