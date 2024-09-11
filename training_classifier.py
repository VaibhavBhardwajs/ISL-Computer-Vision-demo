from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Ensure the 'Data and Model' directory exists
os.makedirs('Data and Model', exist_ok=True)

# Load the data from the correct relative path
data_dict = pickle.load(open(os.path.join('Data and Model', 'data.pickle'), 'rb'))

# Determine the target length (the length of the longest sequence, 84 in this case)
target_length = 84

# Pad shorter sequences with zeros to make all sequences have the same length
padded_data = [d + [0] * (target_length - len(d)) if len(d) < target_length else d for d in data_dict['data']]

# Convert to a NumPy array
data = np.asarray(padded_data)

# Convert numeric labels to strings before fitting the model
labels = np.asarray(data_dict['labels'])

# Use LabelEncoder to convert string labels to numeric labels
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)

# Split the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(data, numeric_labels, test_size=0.2, shuffle=True, stratify=numeric_labels)


# Define class weights manually based on observed label distribution
class_weights_dict = {
    'I': 0.8, 'V': 0.6, 'O': 0.7,  
    'C': 0.8, 'A': 0.8, 
    '8': 2.1, 'L': 0.8, 'Z': 0.8, '9': 1.9,  
    '1': 1.9, '3': 1.9, 'U': 0.8, '4': 2.3, 
    'Y': 0.8, 'X': 0.8, '5': 2.2, 'K': 0.8, 
    '7': 1.9, 'N': 0.8, 'R': 0.8, '2': 2.3, 
    'E': 0.8, 'H': 0.8, 'D': 0.8, 'B': 0.8, 
    'T': 0.8, 'F': 0.8, 'W': 0.8, 'M': 0.8, 
    'J': 0.8, '6': 2.1, 'G': 0.8, 
    'P': 0.8,  
    'S': 0.5,  
    'Q': 0.8   
}

# Convert label weights to indexed format (as required by the model)
class_weights_indexed = {i: class_weights_dict[label] for i, label in enumerate(np.unique(labels))}

# Train with RandomForestClassifier, passing the class weights
model = RandomForestClassifier(class_weight=class_weights_indexed)
model.fit(x_train, y_train)

# Predict on the test set
y_predict = model.predict(x_test)

# Evaluate accuracy
score = accuracy_score(y_predict, y_test)
print('\n{:.2f}% of samples were classified correctly! '.format(score * 100))

# Save the model and LabelEncoder in pickle format in the 'Data and Model' directory
with open(os.path.join('Data and Model', 'model.p'), 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, f)
