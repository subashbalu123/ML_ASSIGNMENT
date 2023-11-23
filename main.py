import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sentence_transformers import SentenceTransformer

# Load the dataset
df = pd.read_csv('articles.csv')

# Assuming 'Text' is the input feature and 'Article_Type' is the target
X = df['Text']
y = df['Article_Type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Vectorization using SentenceBERT
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6')
X_train_vec = sbert_model.encode(X_train)
X_test_vec = sbert_model.encode(X_test)

# Step 3: Select and train an ML classifier model
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# Step 4: Hyperparameter tuning (if needed)

# Step 5: Validate and evaluate accuracy using cross-validation
scores = cross_val_score(classifier, X_train_vec, y_train, cv=5)
print(f'Cross-Validation Accuracy: {scores.mean()}')

# Step 6: Present performance metrics on the test set
y_pred = classifier.predict(X_test_vec)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Test Set Accuracy: {accuracy}')
print(metrics.classification_report(y_test, y_pred))

# Step 7: Save and reload the model
from joblib import dump, load
dump(classifier, 'classifier_model.joblib')

# Reload the model
reloaded_classifier = load('classifier_model.joblib')

# Step 8: Extract the heading and Full_article from unknown_articles.csv
unknown_df = pd.read_csv('unknown_articles.csv')

# Assuming 'article_url' is the column name in unknown_df
unknown_urls = unknown_df['article_url']

# Extract headings and full articles using web scraping or other methods

# Step 9: Predict the "Article_Types" for the extracted data
unknown_data =unknown_df['Heading']+unknown_df['Full_Articles']
unknown_data_vec = sbert_model.encode(unknown_data)
predicted_types = reloaded_classifier.predict(unknown_data_vec)


unknown_df['Predicted_Article_Type'] = predicted_types

# Save the updated unknown_articles.csv
unknown_df.to_csv('unknown_articles_with_predictions.csv', index=False)