# Suicide and Depression Prediction

## 📌 Project Overview
This project implements multiple machine learning algorithms to predict whether a person is suicidal or not based on textual data. We experimented with six models: 
- **Naive Bayes Classifier**
- **Random Forest**
- **Decision Tree**
- **Support Vector Machine (SVM)**
- **Gradient Boosting**
- **K-Nearest Neighbors (KNN)**

We compare their accuracy and performance to determine the best model.

## 📂 Dataset Description
We used the **"Suicide and Depression Detection"** dataset from Kaggle, which contains posts from Reddit’s "SuicideWatch" and "depression" subreddits. The dataset was collected using the Pushshift API and initially contained **232,074** texts.

Due to memory constraints, we randomly sampled **5,000** entries for model training and testing. The dataset has three columns:
- **Unnamed column** (index)
- **Text column** (Reddit post content)
- **Class column** (labels indicating suicide risk)

## 🛠 Libraries Used
- **Natural Language Toolkit (NLTK)** – Text processing (tokenization, stemming, stopword removal)
- **Pandas** – Data manipulation
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn (sklearn)** – Machine learning models & evaluation metrics
- **XGBoost** – Gradient boosting classifier

## 🔄 Data Preprocessing
1. **Word Tokenization**: Splitting text into individual words
2. **Lowercasing**: Converting all text to lowercase
3. **Punctuation Removal**: Removing non-word characters
4. **Stop Words Removal**: Eliminating commonly used words (e.g., "the", "is", "and")
5. **Stemming**: Reducing words to their root form (e.g., "running" → "run")
6. **Concatenation**: Reconstructing processed text back into a string

## 📊 Machine Learning Models Implemented
1. **Naive Bayes Classifier**
2. **Random Forest**
3. **Decision Tree**
4. **Support Vector Machine (SVM)**
5. **Gradient Boosting**
6. **K-Nearest Neighbors (KNN)**

### 🔢 Train-Test Split
- **80% Training Data**
- **20% Testing Data**

### 📝 Feature Extraction
We used **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to convert text data into numerical features. The dataset was transformed into **840-dimensional** vectors.

## 📈 Model Performance Comparison
| Model                | F1 Score |
|----------------------|----------|
| Naive Bayes         | 0.86     |
| Random Forest       | 0.799    |
| Decision Tree       | 0.725    |
| SVM                | 0.884    |
| Gradient Boosting   | 0.803    |
| K-Nearest Neighbors| 0.845    |

📌 **Support Vector Machine (SVM) achieved the highest F1 score of 0.884**, making it the best-performing model for this task.

## ✅ Conclusion
- **SVM outperformed other models** due to its ability to handle high-dimensional text data efficiently.
- **Naive Bayes performed well**, but SVM captured complex relationships better.
- **Random Forest was more robust than Decision Tree** due to reduced overfitting.
- **Gradient Boosting had high training accuracy but slightly overfitted on the small dataset.**

1. Run the Jupyter Notebook or Python script to train and evaluate models.

## 📜 License
This project is for educational purposes only and follows ethical AI practices. Please use responsibly.

---
Feel free to contribute, raise issues, or suggest improvements!
