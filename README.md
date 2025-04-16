# 🎬 IMDB Sentiment Classification with XGBoost, LightGBM, and CatBoost

This project explores the classification of movie reviews as positive or negative using the IMDB Movie Rating dataset. It combines **gradient boosting classifiers** with **unsupervised feature extraction techniques** to understand the trade-offs between accuracy and computational efficiency in Natural Language Processing (NLP) tasks.

---

## 💡 Problem Statement

The goal is to classify IMDB movie reviews into positive or negative sentiment using traditional gradient boosting classifiers and evaluate the impact of **feature extraction** techniques like:

- **Self-Organizing Maps (SOM)**
- **Restricted Boltzmann Machine (RBM)**
- **LSTM-based Autoencoder**

---

## 📦 Project Components

### 🔍 1. Dataset
The project uses the [IMDB Movie Review Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), consisting of 50,000 labeled movie reviews for sentiment classification (25k positive, 25k negative).

### 🧠 2. Feature Extraction Methods

- **SOM (Self-Organizing Maps):** A type of unsupervised neural network used to reduce dimensionality and visualize high-dimensional data by clustering similar reviews.
  
- **RBM (Restricted Boltzmann Machine):** A shallow, two-layer neural network that learns a probability distribution over its set of inputs.

- **LSTM-based Autoencoder:** An encoder-decoder network using Long Short-Term Memory layers to capture temporal dependencies and compress the review text into latent representations.

### 🤖 3. Classifiers Used

We trained three powerful gradient boosting models:

- **XGBoost**
- **LightGBM**
- **CatBoost**

These were applied on:
- The raw dataset (without feature extraction)
- The dataset after applying SOM, RBM, and Autoencoder transformations

---

## 🧪 Experiment Workflow

1. **Preprocessing:**
   - Cleaned and tokenized IMDB reviews
   - Converted text to numerical format (TF-IDF/Embedding)

2. **Model Training:**
   - Trained XGBoost, LightGBM, and CatBoost on raw and feature-extracted datasets

3. **Feature Extraction:**
   - SOM, RBM, and Autoencoder applied separately to extract lower-dimensional features
   - Models were retrained on transformed features

4. **Evaluation:**
   - Compared models based on **Accuracy** and **Execution Time**

---

## 📊 Results Summary

| Classifier | Dataset Type       | Accuracy | Time (s) |
|------------|--------------------|----------|----------|
| XGBoost    | Raw                | 85.26%   | 94.6     |
| LightGBM   | Raw                | 85.78%   | 30.99    |
| CatBoost   | Raw                | 86.36%   | 533.85   |
| XGBoost    | SOM                | 56.00%   | 5.166    |
| LightGBM   | SOM                | 57.92%   | 7.266    |
| CatBoost   | SOM                | 56.10%   | 3.52     |
| XGBoost    | RBM                | 52.33%   | 2.61     |
| LightGBM   | RBM                | 54.98%   | 2.78     |
| CatBoost   | RBM                | 51.34%   | 20.74    |
| XGBoost    | Autoencoder (LSTM) | 61.72%   | 1.07     |
| LightGBM   | Autoencoder        | 61.29%   | 0.95     |
| CatBoost   | Autoencoder        | 61.58%   | 18.81    |

---

## 🧠 Insights & Conclusion

- **CatBoost** gave the highest accuracy on the raw dataset but was the slowest.
- **LightGBM** offered the best balance between accuracy and speed.
- **Feature extraction methods**, while reducing execution time, led to significant drops in accuracy.
- **LSTM-based Autoencoders** performed better than SOM or RBM, highlighting their ability to capture temporal structure in text data.

This project highlights the **trade-off between execution time and model performance** and showcases the value of ensemble methods in NLP classification tasks.

---
