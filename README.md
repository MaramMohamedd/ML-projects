# Flower classification using Neural Networks and SVM 
in this project me and my teammates worked on iris data set for flower type classification based on patel length , withth , sebal length and sepal width 
we applyed our knowledge in machine learning starting from data preprocessing to model evaluation , we also used neural networks implemented with keras, svm with 3 different kernals  , feature scaling  , classification report to discover more about the flase negative and false positive , confusion matrix  


## Data Preprocessing
Loading the dataset: We used the Iris dataset, which is available in scikit-learn. It contains 150
samples of Iris flowers, each with 4 features and a target variable that classifies the species into
three classes
```bash
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
```

manually encoded the target column using map which maps each target name into a value 
```bash
df['target'] = df['target'].map(dict(zip(range(3), iris.target_names)))
```


## Data exploration
1. View Basic Info About the Dataset
2. Check for missing values (no missing values)

```bash 
print(df.info())
print(df.isnull().sum())
```

## Splitting the data:
We split the data into 70% training and 30% testing. We used stratify=y to
make sure each class is equally represented in both sets
X = df.drop('target', axis=1) # Features
y = df['target']  # Target variable

```bash
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```


## Feature Scaling/Normalization using Standard Scaler 
We used StandardScaler to normalize the data.
why?
● Features like "sepal length" and "petal width" have different scales.
● Without normalization, the SVM might give more importance to larger values.
● StandardScaler makes all features have a mean = 0 and standard deviation = 1, so
they’re on the same scale.
```bash
# normalizing the data 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Visualzing the data Using Pair plot 
from pair plot we noticed that the sesota class is easily seprable and don't overlap like versicolor and verginica classes 
```bash 
# Pairplot
sns.pairplot(df, hue='target')
plt.show()
```

Also used correlation matrix to see corr between features 
actually pair plot also showing feature correlations but in a way that can be observed using visuializing but corr matrix showing corr in a numerical way 
both helped us to determine what to keep and what to drop , we noticed that the 4 features are important 

```bash 
# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()
```

## SVM Implementation
How SVM Works:SVM is a supervised learning algorithm used for classification. It tries to find
the best boundary (called a hyperplane) that separates the different classes in your data
### Why SVM Is Good:
● Works well with small to medium datasets
● Handles non-linear boundaries using kernels
● Often accurate and robust
Step 1: Import SVM and Metrics + Ensure encoding instead of manually 
we tried to experience diff types of encoding 
```bash
#encoding from strings to integers 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded  = le.transform(y_test)
```
```bash
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
```

### Step 2: Try Different Kernels
We tested 3 SVM kernels:
● linear: A straight decision boundary
● poly: Polynomial curves
● rbf: Gaussian (smooth curved) boundaries
Using Grid search to get the best parameters 
```bash
# Initialize kernels and parameter grid
kernels = ['linear', 'poly', 'rbf']
param_grid = {
    'linear': {'C': [0.1, 1, 10]},
    'poly': {'C': [0.1, 1, 10], 'degree': [2, 3], 'gamma': ['scale', 'auto']},
    'rbf': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
}

best_params = {}
results = {}

for kernel in kernels:
    print(f"\n Grid Search for Kernel: {kernel}")
    svm = SVC(kernel=kernel)
    
    # Perform Grid Search
    clf = GridSearchCV(svm, param_grid[kernel], cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train_encoded)
    
    # Get best parameters
    best_params[kernel] = clf.best_params_
    print(f"Best Parameters: {best_params[kernel]}")
```
## Training the model with SVM 
```bash
   # Train model using best parameters
    best_svm = SVC(kernel=kernel, **best_params[kernel])
    best_svm.fit(X_train, y_train_encoded)
 ```
## Testing 
```bash
    # Predict on training and testing data
    y_train_pred = best_svm.predict(X_train)
    y_test_pred = best_svm.predict(X_test)
    
    train_acc = accuracy_score(y_train_encoded, y_train_pred)
    test_acc = accuracy_score(y_test_encoded, y_test_pred)
   ```

 ## Results 
 ```bash
    # Train model using best parameters
    best_svm = SVC(kernel=kernel, **best_params[kernel])
    best_svm.fit(X_train, y_train_encoded)
    
    # Predict on training and testing data
    y_train_pred = best_svm.predict(X_train)
    y_test_pred = best_svm.predict(X_test)
    
    train_acc = accuracy_score(y_train_encoded, y_train_pred)
    test_acc = accuracy_score(y_test_encoded, y_test_pred)
     plt.show()
   ```
1. Linear Kernel
● Accuracy: 91.11%
● Performs well overall, especially on Setosa and Versicolor.
● Slight misclassification in Virginica

2. Polynomial Kernel
● Accuracy: 86.67%
● Performs best on Setosa, but struggles with Virginica (some overlap with Versicolor).
● Slightly lower performance than linear and RBF

3. RBF Kernel
● Accuracy: 93.33% Best performance
● Good balance across all three classes.
● Misclassified only a few samples in Versicolor and Virginica.

## Neural Networks implementation using Keras 
Firstly we imported the necessary libraries and to import keras we have to import
Tensorflow first and to construct the model we need the following :
Layers , denses (edges also contain the units , neurons , input shape , activation
functions and so on ..)
```bash
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```
## 1-Preparing data
We needed only to convert the target Y , so we used label encoding to convert labels
from strings to integers then used one hot encoding to be easy for the model to work
with .
we encoded using LabelEncosding before 
Then one hot encoding from keras to_categorical
Also we got the number of classes using ```len(set(y))``` as the set contains only the
unique values = three classes
```bash
#converting into one hot
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train_encoded, num_classes=num_classes)
y_test = to_categorical(y_test_encoded, num_classes=num_classes)
```

## 2-Define the Model:
From my point of view this was the most important and interesting part as i learned a
lot while modifying the layers and the comments indicates our trials
The process is like the following 1- inputs get into the neuron 2- keras initialize the
weights and do the sum in the neuron then applying the activation function we
specified so the first layer means input layer + 1’st hidden layer
Then modifying the layers as if the model needs to be more complex or not , we played
in the units = number of neurons for each layer also the number of layers , also we used
activation function softmax as it produces probabilities for each class and takes the
highest probability as the chosen class
```bash
model = Sequential([
 Dense (64, input_shape=[4] , activation='relu') ,
 Dense (32 , activation='relu') , # we tried first with simple number
of units= 24 , 12 and acc = 69%
 #then we wanted to improve the acc and reducing the loss so we made the
model more complex
 # so we added another hidden layer and units = 64 , 32 the acc = 86%
then we added more layers and units to improve the acc
 #but the acc reduced so our data needs less complexity
 #Dense (16 , activation='relu') ,
 Dense (3, activation='softmax')
])
#we will try to use early stopping to prevent over fitting
# Add early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10,
restore_best_weights=True)
```
Also we needed to use early stopping to address over fitting in the model as the model
while training gave me a validation acc = 100%


## 3-Compile the Model:
To compile the model we used adam optimizer as it worked perfectly with us in
optimizing and the loss was categorical_cross_entropy as we worked with a
classification project also to evaluate the model while training phase we used metrics =
[‘accuracy’]
```bash
model.compile(optimizer = 'adam' , loss='categorical_crossentropy' ,
metrics = ['accuracy'] )
```
## 4-Training the model and hyperparameter tuning:
In the hyperparameter tuning we played with the number of layers , epochs and the
batch size and in the comments we indicated our trials.
There is also a small note : in the number of epochs we put it in a variable as we used it
later to visualize the loss across number of epochs
```bash
epochs = 50#making it in a variable because i will use it latter in
visualizing
history = model.fit( X_train ,y_train , validation_split=0.2, batch_size
= 12
 , epochs= epochs , verbose = 1,
 callbacks =[early_stop])
#we played also in the number of epochs and the batch size so we tried at
first the number of epochs = 10
#and the batch size = 64 the acc = 85 % at max
#we reduced the batch size into 12 only and increased the number of epochs
to 50
# and the acc improved to 98% (maybe it overfits so we will tets it to
know and tune)
#after evaluating we found the trainig acc = 98% and the testing is 91 %
abit overfitting
# since we said previously we will use early stopping so we will increase
the number of epochs
```
## 5-Evaluating the model and see whether if it overfits or not
```bash
test_loss , test_acc= model.evaluate(X_test , y_test)
print(f'the test accuracy :{test_acc*100:.2f}%')
```
## 6-Making predictions to get the classification report and the confusion matrix:
We reversed the one hot encoding back
```bash
from sklearn.metrics import classification_report, confusion_matrix
# Get true labels (reverse one-hot encoding)
y_true = np.argmax(y_test, axis=1)
# Get predicted labels for test set
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
```
## Confusion Matrix
```bash
rose_palette = sns.light_palette("lightcoral", as_cmap=True)
cm = confusion_matrix(y_true , y_pred)
sns.heatmap(cm ,annot=True , cmap=rose_palette , xticklabels=le.classes_,
yticklabels=le.classes_)
plt.ylabel('Actual') #actual
plt.xlabel('Predicted')
plt.show()
```
## The Training and validation loss graph through epochs :
Firstly we saved the model in variable named history to use it now to get the history of
the model we did ```.history()```
```bash
train_loss= history.history['loss']
val_loss= history.history['val_loss']
```
Also we saved the number of epochs in variable epochs which enabled us to use itin
visualizing
```bash
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.show()
```
It decreased then stabilized also we tried in the hyperparameter tuning phase
increasing the number of epochs and the acc decreased so the graph is realistic and we
can rely on it
