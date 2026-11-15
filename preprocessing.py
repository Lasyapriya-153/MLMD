
# preprocessing.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# 1. Generate sample data
def generate_data():

    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        random_state=42
    )

    df = pd.DataFrame(X, columns=['age', 'income'])

    # Add categorical features
    df['city'] = ['Hyderabad', 'Mumbai', 'Chennai', 'Delhi'] * 25
    df['occupation'] = ['Student', 'Engineer', 'Teacher', 'Doctor'] * 25

    # Introduce some missing values
    df.loc[0, 'age'] = None
    df.loc[1, 'city'] = None

    return df, pd.Series(y, name='target')


# 2. Create preprocessing pipeline
def create_pipeline():

    numeric_features = ['age', 'income']
    categorical_features = ['city', 'occupation']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant',
                                  fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


# 3. Execute preprocessing
if __name__ == "__main__":

    # Generate dataset
    X, y = generate_data()

    print("Original Dataset:")
    print(X.head())

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Create pipeline
    preprocessor = create_pipeline()

    # Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform testing data
    X_test_processed = preprocessor.transform(X_test)

    print("\nPreprocessing completed successfully!")

    print("Original training shape:", X_train.shape)
    print("Processed training shape:", X_train_processed.shape)

    print("Original testing shape:", X_test.shape)
    print("Processed testing shape:", X_test_processed.shape)
