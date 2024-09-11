# train classifier

# imports 
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

# for i, d in enumerate(data_dict['data']): #to check length
#     print(f"Data index {i} has length: {len(d)}")

# for i, label in enumerate(data_dict['labels']): 
#     print(f"Label index {i}: {label}")


# Determine the target length (the length of the shortest sequence, 42 in this case)
# target_length = 42

# Trim longer sequences to make all sequences have the same length
# trimmed_data = [d[:target_length] for d in data_dict['data']]

# Determine the target length (the length of the longest sequence, 84 in this case)
target_length = 84

# Pad shorter sequences with zeros to make all sequences have the same length
padded_data = [d + [0] * (target_length - len(d)) if len(d) < target_length else d for d in data_dict['data']]

# Convert to a NumPy array
data = np.asarray(padded_data)


# Convert to a NumPy array
# data = np.asarray(trimmed_data)

# Convert numeric labels to strings before fitting the model
labels = np.asarray(data_dict['labels'])

# spliting the data into training and testing 

x_train,x_test,y_train,y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train,y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict,y_test)

# Printing the accuracy with two decimal places
print('\n{:.2f}% of samples were classified correctly! '.format(score * 100))

# Saving the model in pickle format in the 'Data and Model' directory
with open(os.path.join('Data and Model', 'model.p'), 'wb') as f:
    pickle.dump({'model': model}, f)