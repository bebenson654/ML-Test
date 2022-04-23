from sklearn import datasets
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# data source path. comment out the one you don't want to use
# data_path = ".\\Balanced\\Train"
data_path = ".\\Unbalanced\\Train"

categories = ["ham", "spam"]

# loading datasets
data_train = datasets.load_files(data_path, encoding="utf-8", categories=categories)
data_test = datasets.load_files(data_path, encoding="utf-8", categories=categories)

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(data_train.data)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_test_counts = count_vect.transform(data_test.data)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# comment out the model you don't want to use
# model = svm.SVC()
model = MultinomialNB()

model.fit(X_train_tfidf, data_train.target)
predicted = model.predict(X_test_tfidf)


# Evaluation of prediction
print(
    "\n"
    + metrics.classification_report(
        data_test.target, predicted, target_names=data_test.target_names
    )
    + "\n"
)

print(metrics.confusion_matrix(data_test.target, predicted))
