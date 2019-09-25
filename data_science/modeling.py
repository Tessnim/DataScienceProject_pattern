from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit(X_train,y_train)
#clf.fit(train_x, train_y, early_stopping_rounds=20, eval_set=[(test_x, test_y)])
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
print(" xgboost: %.2f%%" %(acc_test*100))