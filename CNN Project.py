import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_and_preprocess_data(directory):
    images = []
    labels = []

    for label, fruit in enumerate(os.listdir(directory)):
        fruit_path = os.path.join(directory, fruit)
        
        for filename in os.listdir(fruit_path):
            if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                img_path = os.path.join(fruit_path, filename)
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Unable to read image. Skipping...")
                        continue
                    
                    img = cv2.resize(img, (50, 50))  # Resize image to desired dimensions
                    img = img / 255.0  # Normalize pixel values
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing image:")

    return np.array(images), np.array(labels)

# Load and preprocess data
directory = "D:\\GVSU\\CIS 378\\data\\fruits"
images, labels = load_and_preprocess_data(directory)

# Split data into training+validation and testing sets (85% train+val, 15% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(images, labels, test_size=0.15, random_state=42)

# Split the training+validation set into actual training and validation sets (60% train, 25% val of total data)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.294, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(150, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
optimizer = Adam(learning_rate= 0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=32,  validation_data=(X_val, y_val))

# Evaluate model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Calculate accuracy for each class
class_accuracy = {}
for class_index in range(10):
    class_samples = X_test[y_test == class_index]
    class_labels = y_test[y_test == class_index]
    class_loss, class_acc = model.evaluate(class_samples, class_labels, verbose=0)
    class_accuracy[class_index] = class_acc

class_names = ['apple', 'banana', 'blueberry', 'cherry', 'grapes', 'kiwi', 'mango', 'orange', 'stawberry', 'watermelon']

# Print accuracy for each class
for class_index, acc in class_accuracy.items():
    print(f"Accuracy for class {class_names[class_index]}: {acc}")

# Select one sample image from each class for testing
X_sample_test = []
y_sample_test = []

for class_index in range(10):
    class_samples = X_test[y_test == class_index]
    if len(class_samples) > 0:  # Ensure there is at least one sample per class
        X_sample_test.append(class_samples[0])
        y_sample_test.append(class_index)

X_sample_test = np.array(X_sample_test)
y_sample_test = np.array(y_sample_test)

# Predict labels for the selected test images
predictions = model.predict(X_sample_test)

# Display the selected test images along with predicted labels
for i in range(len(X_sample_test)):
    plt.figure(figsize=(4, 4))  # Adjust figure size as needed
    plt.imshow(X_sample_test[i])
    plt.title(f"True: {class_names[y_sample_test[i]]}, Predicted: {class_names[np.argmax(predictions[i])]}")
    plt.axis('off')
    plt.show()
