import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn. datasets import make_classification
from sklearn.imputer i,mport SimpleImputer
fromn sklearn.pipeline import Pipeline
from sklearn.preprocessing import train_test_split , StandardScaler
def create_pipeline:
       numerical_features=['age','income']
       categorical_features=['city','occupation']
       numeric_transformer=Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),('scaler', StandardScaler())])
       categorical_transformer=Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing'))('onehot',OneHotEncoder(handle_unknown='ignore'))])
       preprocessor= ColumnTransformer(transfprmers=[('num',numeric_transformer,numeric_features),('cat', categorical_transformer, categorical_features)])
       return preprocessor
if_name_=="_main_":
   print("Preprocessing  pipeline template ready.")
        
                                       
                            
                                                                                                                  
