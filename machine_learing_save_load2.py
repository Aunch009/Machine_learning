from sklearn.externals import joblib
lr = joblib.load('Advertising.joblib')  #load 'Advertising.joblib' for using
tv=float(input('tv='))#need to debug in terminal
radio=float(input('Radio='))
new=float(input('New='))
print(lr.predict([[tv,radio,new]]))
