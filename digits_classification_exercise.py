
print(__doc__)

from sklearn import datasets, neighbors, linear_model

sayilar = datasets.load_digits()
x_sayilari = sayilar.data
y_sayilari = sayilar.target

uzunluk = len(y_sayilari)

X_train = x_sayilari[:.9 * uzunluk]
y_train = y_sayilari[:.9 * uzunluk]
x_test = x_sayilari[.9 * uzunluk:]
y_test = y_sayilari[.9 * uzunluk:]

knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression()

print('KNN score: %f' % knn.fit(x_train, y_train).score(x_test, y_test))
print('LogisticRegression score: %f'
      % logistic.fit(x_train, y_train).score(x_test, y_test))
