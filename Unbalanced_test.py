from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

categories = ["ham", "spam"]

data_train = datasets.load_files(
    ".\\Unbalanced\\Train", encoding="utf-8", categories=categories
)
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(data_train.data)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

model = svm.SVC()
model.fit(X_train_tfidf, data_train.target)


data_test = datasets.load_files(
    ".\\Unbalanced\\Test", encoding="utf-8", categories=categories
)
X_test_counts = count_vect.transform(data_test.data)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# for txt, cat in zip(data_test.data, data_test.target):
#     if data_test.target_names[cat] == "spam":
#         print(data_test.target_names[cat] + " => " + txt)

predicted = model.predict(X_test_tfidf)

# for x, cat in zip(data_test.data, predicted):
#     print("%r => %s" % (x, data_train.target_names[cat]))


# Evaluation of prediction
print("Acuracy: " + str(np.mean(predicted == data_test.target)))

print(
    "\n"
    + metrics.classification_report(
        data_test.target, predicted, target_names=data_test.target_names
    )
    + "\n"
)

print(metrics.confusion_matrix(data_test.target, predicted))
