import numpy as np
import pandas as pd

# Data pipeline, scaling, normalizing, etc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Machine Learning Algorithms
from sklearn.mixture import GaussianMixture




def transform_data_for_gaussian_mixtures(pandas_object, field_name):
    
    # Create a new pandas object to temporarily store the data before imputing
    data = pd.DataFrame()

    # Convert the Time variable to Seconds Since Epoch
    data["Seconds Since Epoch"] = pandas_object['Seconds Since Epoch']
    data[field_name] = pandas_object[field_name]    
    
    # Define a pipline to clean numerical data
    # Note that this does NOT rescale values - this function expects it to already be scaled!
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
    ])

    # Run the pipeline
    data_imputed = num_pipeline.fit_transform(data)    
    
    return data_imputed





def get_anomalies_using_gaussian_mixtures(data_imputed, density_threshold_percent):
    # Gaussian Mixtures Parameters
    N_COMPONENTS = 1           # The number of regions to generate - needs to be 1 for this use case
    N_INIT = 10
    COVARIANCE_TYPE = "tied"
    
    gm = GaussianMixture(n_components=N_COMPONENTS, n_init=N_INIT,covariance_type=COVARIANCE_TYPE, random_state=42)
    gm.fit(data_imputed)

    # gm.predict(vector_magnitude_data_imputed)
    # gm.predict_proba(vector_magnitude_data_imputed)
    # gm.score_samples(vector_magnitude_data_imputed)

    densities = gm.score_samples(data_imputed)

    density_threshold = np.percentile(densities, density_threshold_percent)

    anomalies = data_imputed[densities < density_threshold]   
    
    return gm, anomalies