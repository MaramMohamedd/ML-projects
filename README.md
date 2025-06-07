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
