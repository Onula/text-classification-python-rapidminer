import os
import glob
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Λειτουργία για φόρτωση κειμένων από φακέλους και κατηγοριοποίησή τους
def load_data_from_folders(category_folders):
    texts = []
    labels = []
    
    # Για κάθε κατηγορία, φορτώνουμε τα αρχεία .txt και προσθέτουμε τα κείμενα και τις ετικέτες
    for category, folder in category_folders.items():
        for filename in glob.glob(os.path.join(folder, '*.txt')):
            with open(filename, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(category)  # Ετικέτα για την κατηγορία

    return texts, labels

# Γενική συνάρτηση για εκτέλεση Naive Bayes (Multinomial ή Bernoulli) με επιλογή binary ή term occurrences
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

    # Εκπαίδευση του μοντέλου
    start_time = time.time()
    model.fit(X_train_vect, y_train)
    y_pred = model.predict(X_test_vect)
    execution_time = time.time() - start_time

    # Υπολογισμός μετρικών
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, digits=4)  # Ακρίβεια 4 δεκαδικών ψηφίων
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Εκτύπωση αποτελεσμάτων
    print_results(model_name, execution_time, execution_time + execution_time_load_data, accuracy, conf_matrix, class_report)


# Εκτύπωση των αποτελεσμάτων με σαφήνεια
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

# Σταθερές τιμές για κλάσεις και directories
category_folders = {
    'neg': 'txt_sentoken/neg',
    'pos': 'txt_sentoken/pos'
}

# Σταθερές τιμές για test_size και document frequency parameters
test_size = 0.50
min_df = 0.04
max_df = 0.82
random_state=1992

# Φόρτωση δεδομένων
print("\nLoading data...")
start_time = time.time()
texts, labels = load_data_from_folders(category_folders)  # Κλήση της συνάρτησης για φόρτωση δεδομένων
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=random_state)
execution_time_load_data = time.time() - start_time
print(f"Data loaded successfully in {execution_time_load_data:.4f} seconds.\n")

# Εκτέλεση και αξιολόγηση των δύο μοντέλων με Term Occurrences και Binary Term Occurrences
naive_bayes("multinomial",  X_train, X_test, y_train, y_test, min_df, max_df)  # Term Occurrences
naive_bayes("bernoulli",  X_train, X_test, y_train, y_test, min_df, max_df)  # Binary Term Occurrences
