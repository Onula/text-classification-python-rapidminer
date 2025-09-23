# Text Classification with Naive Bayes (Python & RapidMiner)

## 📌 Project Description
This project explores **text classification** using the **Naive Bayes algorithm**, implemented in two different environments:  
- **Python (scikit-learn)**  
- **RapidMiner (Text Processing extension)**  

Both **Multinomial Naive Bayes** (term occurrences) and **Bernoulli Naive Bayes** (binary term occurrences) were tested.  
The aim was to compare their performance in terms of **accuracy, precision, recall, F1-score, and execution time**.

---

## 🛠️ Tools & Technologies
- **Python 3.10+**
- **scikit-learn**
- **RapidMiner**
- **Jupyter Notebook / .py scripts**
- **Word/PDF report with results**

---

## 📂 Repository Structure
```
text-classification-python-rapidminer/
│
├── code/                   # Python scripts and notebooks
├── report/                 # Project report (PDF & DOCX)
├── results/                # Confusion matrices, tables, metrics
└── README.md               # Project documentation
```

---

## 📖 Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Directory Structure](#directory-structure)
5. [Detailed Function Explanation](#detailed-function-explanation)
6. [Usage Instructions](#usage-instructions)
7. [Example Results](#example-results)
8. [Results Summary](#results-summary)
9. [Report](#report)
10. [References](#references)

---

## 🔎 Introduction
This project implements a **text sentiment classifier** using the Naive Bayes algorithm. It includes two variations of Naive Bayes:

- **Multinomial Naive Bayes**: Based on word frequency in texts, suitable for problems where term frequency matters.
- **Bernoulli Naive Bayes**: Uses binary term occurrence (presence/absence of words), suitable for cases where the existence of a word is more important than its frequency.

The classifier can be used for sentiment analysis in text files (e.g., positive or negative sentiment) and can be applied to text documents organized into folders.

---

## ✨ Features
- Support for **Multinomial** and **Bernoulli** Naive Bayes.
- Calculation of performance metrics such as **accuracy**, **confusion matrix**, and **classification report**.
- Parameterization via **min_df** and **max_df** for term frequency pruning.
- Capable of analyzing large text datasets with **fast training and prediction**.

---

## ⚙️ Requirements
To run this project, make sure you have installed the following Python libraries:

- **scikit-learn**
- **glob**
- **os**
- **time**

Install dependencies:

```bash
pip install scikit-learn
```

---

## 📁 Directory Structure
Your project folder should follow this structure:

```
project-directory/
│
├── txt_sentoken/
│   ├── neg/       # Files with negative sentiment
│   └── pos/       # Files with positive sentiment
│
└── main_script.py # The main script containing the code
```

---

## 🔧 Detailed Function Explanation

### load_data_from_folders(category_folders)
Reads text files from specific folders that correspond to categories (e.g., 'neg' and 'pos') and returns the texts along with their labels.

### naive_bayes(nb_type, X_train, X_test, y_train, y_test, min_df, max_df)
Trains a Naive Bayes model (**Multinomial** or **Bernoulli**, depending on `nb_type`) and performs predictions. Data is first vectorized using `CountVectorizer`, then the model predicts and prints the results.

### print_results(model_name, execution_time, execution_time_with_load, accuracy, conf_matrix, classification_rep)
Prints the results of the model execution: execution time, accuracy, confusion matrix, and classification report (precision, recall, f1-score).

---

## ▶️ Usage Instructions
1. **Prepare folders**: Place text files into the correct folders (`neg` and `pos`) as shown above.
2. **Parameter settings**: Adjust term frequency thresholds (`min_df`, `max_df`), training/test split (`test_size`), and other script parameters as needed.
3. **Run**: Execute the script to train the model and view results.

Example execution:
```bash
python main_script.py
```

---

## 📊 Example Results

### Multinomial Naive Bayes
```
Confusion Matrix:
[[234  45]
 [ 32 289]]

Accuracy: 0.8760

Classification Report:
              precision    recall  f1-score   support
         neg     0.8795    0.8396    0.8591       279
         pos     0.8657    0.9000    0.8825       321
```

---

## 📈 Results Summary
- **Bernoulli Naive Bayes** achieved the best overall performance:
  - Python: **79.80% accuracy**
  - RapidMiner: **79.75% accuracy**
- **Multinomial Naive Bayes**:
  - Python: **77.40% accuracy**
  - RapidMiner: **73.40% accuracy**
- Python execution was approximately **twice as fast** as RapidMiner.

---

## 📄 Report
The full academic report with detailed methodology, results, and comparison is available in:  
📂 `report/Text Classification (EN).pdf`

---

## 📜 References
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.  
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.  
- RapidMiner Documentation: [https://docs.rapidminer.com](https://docs.rapidminer.com)  
- Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/naive_bayes.html](https://scikit-learn.org/stable/modules/naive_bayes.html)  

---

✍️  Onour Imprachim
