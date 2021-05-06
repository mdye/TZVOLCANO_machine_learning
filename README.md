## Toward autonomous detection of anomalous GNSS data via applied unsupervised artificial intelligence 

## Authors
Mike Dye, D. Sarah Stamps, Myles Mason

## Abstract
Artificial intelligence applications within the geosciences are becoming increasingly common, yet there are still many challenges to adapt established techniques to geoscience datasets. Applications in the realm of volcanic hazards assessment show great promise. This Jupyter notebook ingests real-time GNSS data streams from the EarthCube CHORDS (Cloud-Hosted Real-time Data Services for the Geosciences) portal TZVOLCANO, applies unsupervised learning algorithms to perform automated data quality control (“anomaly detection”), and explores autonomous detection of unusual volcanic activity. TZVOLCANO streams real-time GNSS positioning data from the active Ol Doinyo Lengai volcano in Tanzania through UNAVCO’s real-time GNSS data services, which provide near-real-time positions processed by the Trimble Pivot system. The positioning data (latitude, longitude, and height) are imported into this Jupyter Notebook in user-defined time spans. The raw data are then collected in sets by the notebook and processed to extract useful calculated variables, such as an average change vector, in preparation for the machine learning algorithms. In order to create usable data, missing time-series points are imputed using the “most frequent” strategy. After this initial preparation, unsupervised K-means and Gaussian Mixture machine learning algorithms are utilized to  locate and remove data points that are likely unrelated to volcanic signals. Preliminary results indicate that both the K-means and Gaussian Mixture machine learning algorithms perform well at identifying regions of high noise within tested GNSS data sets.

## Keywords
TZVOLCANO, CHORDS, UNAVCO, EarthCube, Artificial Intelligence, Machine Learning, unsupervised learning, automated QA/QC, GNSS


## Binder
These notebooks are Binder enabled and can be run on [mybinder.org](https://mybinder.org/) for free using the links below

### EarthCube Notebook
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mdye/TZVOLCANO_machine_learning.git/HEAD?filepath=EC_05_Template_Notebook_for_EarthCube_Long_Version.ipynb)


### Full Repo
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mdye/TZVOLCANO_machine_learning.git/HEAD)



