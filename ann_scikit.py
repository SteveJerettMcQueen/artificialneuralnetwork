import os
import datetime as dt
import numpy as np

from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from util import load_data

################################################################################

# Classification data
target_names=['Female', 'Male']

dir_label = [
    ['badeer-r', 1], ['benson-r', 1], ['blair-l', 0],
    ['cash-m', 0], ['corman-s', 1], ['hain-m', 1]]

dataset = load_data(dir_label)

X = np.array(dataset[0])
y = dataset[1]

# Sci-Kit Learn Artificial Neural Network Classifiers
# Train/Test split model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 20)

# Multi-Layer Percerptron Classifier
# logistic Sigmodial Function & Stochastic Gradient-Based optimizer
mlp = MLPClassifier(activation='logistic', solver='adam', ).fit(X_train, y_train)
y_mlp_pred = mlp.predict(X_test)
acc = metrics.accuracy_score(y_test, y_mlp_pred)
cfm = metrics.confusion_matrix(y_test, y_mlp_pred)
report = classification_report(y_test, y_mlp_pred, target_names=target_names)
    
# Hyperbolic Tangent Function & Stochastic Gradient-Based optimizer
mlp2 = MLPClassifier(activation='tanh', solver='adam', ).fit(X_train, y_train)
y_mlp2_pred = mlp2.predict(X_test)
acc2 = metrics.accuracy_score(y_test, y_mlp2_pred)
cfm2 = metrics.confusion_matrix(y_test, y_mlp2_pred)
report2 = classification_report(y_test, y_mlp2_pred, target_names=target_names)

# Rectified Linear Unit & Stochastic Gradient-Based optimizer
mlp3 = MLPClassifier(activation='relu', solver='adam', ).fit(X_train, y_train)
y_mlp3_pred = mlp3.predict(X_test)
acc3 = metrics.accuracy_score(y_test, y_mlp3_pred)
cfm3 = metrics.confusion_matrix(y_test, y_mlp3_pred)
report3 = classification_report(y_test, y_mlp3_pred, target_names=target_names)

# Write to file
filename = 'dataset/results.txt'
file_exists = os.path.exists(filename)
append_write = 'a' if(file_exists) else 'w'
f = open(filename, append_write)

# Write Data information
f.write("---------------------------------------------------------------------\n")
f.write("Date: " + str(dt.datetime.now().strftime("%m-%d-%Y")) + '\n')
f.write("Data Set: " + str(len(X)) + "\n")
f.write("X Tranining Set: " + str(len(X_train)) + " ; X Test Set: " + str(len(X_test)) + "\n")
f.write("Y Tranining Set: " + str(len(y_train)) + " ; Y Test Set: " + str(len(y_test)) + "\n")
f.write("\n")

# Write Sigmodial Function metrics
f.write("Sigmodial Accuracy: " + "{0:.4f}".format(acc) + "\n")
f.write("Layers: " + str(mlp.n_layers_) + " ; Iterations: " + str(mlp.n_iter_) + " ; " )
f.write("Outputs: " + str(mlp.n_outputs_) + "\n")
f.write("Confusioin Matrix: " + str(cfm.ravel()) + "\n")
f.write("Classification Report:\n" + report + "\n")

# Write Hyperbolic Tangent Function metrics
f.write("Hyperbolic Tangent Accuracy: " + "{0:.4f}".format(acc2) + "\n")
f.write("Layers: " + str(mlp2.n_layers_) + " ; Iterations: " + str(mlp2.n_iter_) + " ; " )
f.write("Outputs: " + str(mlp2.n_outputs_) + "\n")
f.write("Confusioin Matrix: " + str(cfm2.ravel()) + "\n")
f.write("Classification Report:\n" + report2 + "\n")

# Write Rectified Linear Unit Function metrics
f.write("Rectified Linear Unit Accuracy: " + "{0:.4f}".format(acc3) + "\n")
f.write("Layers: " + str(mlp3.n_layers_) + " ; Iterations: " + str(mlp3.n_iter_) + " ; " )
f.write("Outputs: " + str(mlp3.n_outputs_) + "\n")
f.write("Confusioin Matrix: " + str(cfm3.ravel()) + "\n")
f.write("Classification Report:\n" + report3 + "\n")
f.close()
