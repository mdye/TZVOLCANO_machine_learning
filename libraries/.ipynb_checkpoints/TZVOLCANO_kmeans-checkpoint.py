

import pandas as pd

# Data pipeline, scaling, normalizing, etc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

        
        
def transform_data_for_kmeans(pandas_object, field_name):
    
    # Create a new pandas object to temporarily store the data before imputing
    data = pd.DataFrame()

    # Convert the Time variable to Seconds Since Epoch
    data["Seconds Since Epoch"] = pandas_object['Seconds Since Epoch']
    data[field_name] = pandas_object[field_name]
    
    
    # Define a pipline to clean numerical data
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('std_scaler', StandardScaler()),
    ])

    # Run the pipeline
    data_imputed = num_pipeline.fit_transform(data)    
    
    return data_imputed        
       
