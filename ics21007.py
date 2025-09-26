import os
import glob
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Function to load texts from folders and assign categories
def load_data_from_folders(category_folders):
    texts = []
    labels = []
    
    # For each category, load .txt files and add texts and labels
    for category, folder in category_folders.items():
        for filename in glob.glob(os.path.join(folder, '*.txt')):
            with open(filename, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(category)  # Label for the category

    return texts, labels

# General function to run Naive Bayes (Multinomial or Bernoulli) with choice of binary or term occurrences
def naive_bayes(nb_type, X_train, X_test, y_train, y_test, min_df, max_df):
    if nb_type == "multinomial":
        model_name = "Multinomial Naive Bayes with Term Occurrences"
        model = MultinomialNB()
        vectorizer = CountVectorizer(stop_words='english', token_pattern=r'\b\w{3,}\b', lowercase=True, min_df=min_df, max_df=max_df)
    elif nb_type == "bernoulli":
        model_name = "Bernoulli Naive Bayes with Binary Term Occurrences "
        model = BernoulliNB()
        vectorizer = CountVectorizer(binary=True, stop_words='english', token_pattern=r'\b\w{3,}\b', lowercase=True, min_df=min_df, max_df=max_df)
    else:
        raise ValueError("Invalid nb_type! Please use 'multinomial' or 'bernoulli'.")
    
    # Vectorization
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Train the model
    start_time = time.time()
    model.fit(X_train_vect, y_train)
    y_pred = model.predict(X_test_vect)
    execution_time = time.time() - start_time

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, digits=4)  # Accuracy with 4 decimal places
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results
    print_results(model_name, execution_time, execution_time + execution_time_load_data, accuracy, conf_matrix, class_report)


# Print results clearly
def print_results(model_name, execution_time, execution_time_with_load, accuracy, conf_matrix, classification_rep):
    print("="*50)
    print(model_name)
    print("="*50)
    print(f"Execution Time: {execution_time:.4f} seconds (with loading data: {execution_time_with_load:.4f} seconds)\n")
    
    print("Confusion Matrix:")
    print(conf_matrix)

    print(f"\nAccuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(classification_rep)

# Fixed values for classes and directories
category_folders = {
    'neg': 'txt_sentoken/neg',
    'pos': 'txt_sentoken/pos'
}

# Fixed values for test_size and document frequency parameters
test_size = 0.50
min_df = 0.04
max_df = 0.82
random_state = 1992

# Load data
print("\nLoading data...")
start_time = time.time()
texts, labels = load_data_from_folders(category_folders)  # Call the function to load data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=random_state)
execution_time_load_data = time.time() - start_time
print(f"Data loaded successfully in {execution_time_load_data:.4f} seconds.\n")

# Run and evaluate both models with Term Occurrences and Binary Term Occurrences
naive_bayes("multinomial", X_train, X_test, y_train, y_test, min_df, max_df)  # Term Occurrences
naive_bayes("bernoulli", X_train, X_test, y_train, y_test, min_df, max_df)    # Binary Term Occurrences
