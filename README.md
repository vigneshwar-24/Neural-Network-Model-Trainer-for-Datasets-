# <p align="center">Neural Network Model Trainer for Datasets 

## *Introduction* :
The contemporary landscape is witnessing an exponential rise in the availability of datasets, a direct consequence of increased digital engagement. Beyond major corporations, smaller entities actively curate their datasets, complemented by an abundance of openly accessible data strewn across the internet. However, the expertise required to adeptly construct deep learning models remains scarce, creating a bottleneck for organizations in leveraging their datasets effectively. This underscores the need for skilled deep learning developers capable of crafting intricate code for dataset analysis and model construction.

Amidst the surge in deep learning algorithm usage, a pivotal process involves training models to the specific application's data. This involves iterative analysis, where the Deep Learning Model undergoes forward and backward propagation to adjust its weight parameters, refining accuracy and predictive capabilities during training. Leveraging datasets, the neural network model—such as Convolutional Neural Network and sequential models—can discern intricate relationships and predict sequential data with heightened accuracy.

Hence, my project revolves around engineering a neural network model endowed with the autonomy to train provided datasets independently. Not only will this model generate the deep learning model, but it will also offer users the capability to download it for future applications. This initiative specifically targets smaller organizations and newcomers, providing them with accessible, pre-designed deep learning models finely attuned to their unique datasets.

  
## _Purpose of the Project_ :

The project's purpose is to democratize deep learning model creation by developing an autonomous neural network capable of training on provided datasets. It addresses the challenge of limited expertise in constructing such models, particularly for smaller organizations and newcomers. By leveraging this solution, users gain access to finely-tuned deep learning models tailored to their unique datasets, facilitating enhanced predictive capabilities and accuracy. The initiative aims to empower organizations by offering accessible, pre-designed models that can be downloaded and utilized for various applications. Ultimately, it seeks to unlock opportunities for innovation and problem-solving, even in the absence of specialized expertise, thus fostering broader adoption of deep learning technology.
  
## _Objective_ :
The project aims to develop an autonomous neural network capable of training on provided datasets, addressing the scarcity of expertise in crafting deep learning models. It seeks to democratize access to sophisticated deep learning technology, particularly targeting smaller organizations and newcomers. The objective is to empower users by providing them with pre-designed, finely-tuned deep learning models tailored to their unique datasets. By offering accessible solutions, the project aims to facilitate the adoption and utilization of deep learning technology, enabling organizations to leverage their data effectively for enhanced predictive capabilities and problem-solving. Ultimately, the project aims to foster innovation and enable broader participation in the field of deep learning.
  
## _Abstract_ :

This innovative project introduces a deep learning model that trains on diverse datasets and delivers the trained TensorFlow model to users. It seamlessly combines dynamic dataset provision, model training, and immediate model delivery. Users can adjust parameters like epochs and batch size for tailored machine learning domains, including classification and regression. Transparent logs and performance metrics offer insights into the training process. Prioritizing scalability, user experience, and security, it simplifies model deployment for both non-coders and experienced users. This inclusive solution enhances accessibility to AI models, benefiting the machine learning community by facilitating effortless access and deployment.

## _Methodology_ :
  + The implementation follows a streamlined process for user convenience and adaptability.
  
  - Users define dataset type: regression, image classification, etc.
  
  * System guides users through dataset column selection.

  - Specialized algorithms engage in model training based on selected dataset column.

  - Neural network learns intricate patterns and establishes dataset connections.

  - Upon training completion, system generates downloadable .h5 file.
  - File contains acquired knowledge and insights from neural network training.
  - Users can download and retain file for future use.
  - Trained neural network can be applied in various scenarios or incorporated into applications.
  - Approach promotes flexibility, usability, and efficiency in training tailored neural networks.
  
## _Project FlowChart :
<img width="532" alt="image" src="https://github.com/vigneshwar-24/Neural-Network-Model-Trainer-for-Datasets-/assets/77089276/559e4152-9c01-42fb-aae0-04e15201be68">


## Algorithm :

1. Define dataset type and characteristics (e.g., regression, image classification).
2. Select appropriate neural network architecture (e.g., Convolutional Neural Network, sequential models).
3.Initialize neural network model with random weights.
4. Split dataset into training and testing sets
5. Loop through training data:(forward propogation, caluculate loss,backward propogation)
6. Validate model performance using testing data.
7. If performance is satisfactory:(Save trained model parameters,Generate downloadable file (e.g., .h5 format) containing trained model. )
8. Provide option for users to download the trained model for future applications.
  
## _Program_ :
### main.py
```python
q=int(input("What you wanna do?\n 1.Image-Classification\n 2.Classification\n 3.Regression"))
if q==1:
    exec(open('image-classification.py').read())
elif q==2:
    exec(open('classification.py').read())
else:
    exec(open('regression.py').read())
```
 
  
### Image_classification.py
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Define the image data generator
datagen = ImageDataGenerator(rescale=1./255)
# Specify the path to your dataset
dataset=input("Enter your dataset name?\n")
train_data_dir = dataset
# Set the parameters for the generator
batch_size = 32
img_height = 224
img_width = 224
# Create the generator for training data
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # Use 'categorical' for multiple classes
)
# Retrieve the number of classes from the generator
num_classes = train_generator.num_classes
# Build the model
model = tf.keras.models.Sequential([
    # Add your desired layers here
    tf.keras.layers.Flatten(input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model using the generator
model.fit(train_generator, epochs=10)
model.save("image.h5")
```
### Classification.py
```python
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from keras.callbacks import ModelCheckpoint, EarlyStopping
# Load the dataset
data = pd.read_csv("data.csv")
data = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7,8]]
data = data.replace(" NAN", pd.NA)
data.dropna(inplace=True)
tobepredicted=input("Enter the column name to be predicted?")
label_encoder = LabelEncoder()
data[tobepredicted] = label_encoder.fit_transform(data[tobepredicted])
X = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]]
Y = data[tobepredicted]
if X.isnull().sum().any():
    print("Dataset contains missing values. Please handle them.")
# Data preprocessing
sc = StandardScaler()
X = sc.fit_transform(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
y_train = y_train.astype(int)
y_test = y_test.astype(int)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# Compile the model with binary cross-entropy loss
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')
model.save("classification.h5")
```

### Regression.py
```python
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#csv=input("Name of the CSV file")
boston_data = pd.read_csv("Boston.csv")
# Extract features (X) and target variable (y)
tobepredicted=input("The column name to be predicted ?")
X = boston_data.drop(tobepredicted, axis=1)  # Replace 'target_column_name' with the actual column name
y = boston_data[tobepredicted]  # Replace 'target_column_name' with the actual column name
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Build the regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1)  # Output layer with a single neuron for regression
])
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=0)
# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test)
print(f'Mean Absolute Error on Test Set: {mae}')
# Save the trained model if needed
model.save('model.h5')

```
## Output:
### Datasets
<img width="534" alt="image" src="https://github.com/vigneshwar-24/Neural-Network-Model-Trainer-for-Datasets-/assets/77089276/e023bab8-6bfe-4a2f-be4a-eef9056d52fc">

### Enter the dataset name
<img width="531" alt="image" src="https://github.com/vigneshwar-24/Neural-Network-Model-Trainer-for-Datasets-/assets/77089276/167cf43a-1fd1-4bcf-9e2a-87109ea7d381">

### Saved model and  .h5 file
<img width="533" alt="image" src="https://github.com/vigneshwar-24/Neural-Network-Model-Trainer-for-Datasets-/assets/77089276/b6b50726-8b58-4a03-99d4-fdcbe24a9c7d">

### Saved the model
<img width="277" alt="image" src="https://github.com/vigneshwar-24/Neural-Network-Model-Trainer-for-Datasets-/assets/77089276/25fed40f-ee58-4ea6-a3a3-53de2bc309ef">


## _Conclusion_ :
The "Neural Network Model Trainer for Dataset" enhances machine learning accessibility by prioritizing user engagement. It begins by prompting users to specify their dataset type, streamlining subsequent steps. Systematically guiding users, it collects essential dataset column details vital for neural network training. Specialized algorithms impart knowledge during training, enabling the network to decipher intricate patterns. This fosters the establishment of meaningful data connections. Upon completion, the system generates a downloadable .h5 file, encapsulating enriched insights of the neural network. Serving as a repository of acquired knowledge, this file facilitates future applications and integrations, ensuring user convenience and empowerment in machine learning endeavors.

## _Results_ :

The project aims to democratize deep learning model construction by developing an autonomous neural network capable of independent dataset training. It addresses the scarcity of expertise in constructing such models, particularly for smaller organizations and newcomers. The objective is to empower users by providing accessible, pre-designed deep learning models tailored to their unique datasets. This initiative simplifies model generation and offers users the flexibility to download and apply the trained model for various applications. By targeting accessibility and usability, the project aims to unlock opportunities for innovation and problem-solving in the machine learning landscape, fostering broader participation and utilization of deep learning technology.
