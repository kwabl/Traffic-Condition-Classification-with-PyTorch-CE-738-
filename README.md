# Traffic-Condition-Classification-with-PyTorch-CE-738-

Class assignment for CE 738: a multi-class classifier that predicts traffic class (0, 1, 2) from traffic and weather measurements, built with a single-layer neural network (softmax regression) in PyTorch.

Dataset

transportation.csv has 30 records, 10 in each class.

Column	Description
traffic_volume_vph	Traffic volume (vehicles per hour)
occupancy_pct	Detector occupancy (%)
average_speed_mph	Average speed (mph)
rainfall_mm	Rainfall (mm)
travel_time_min	Travel time (minutes)
traffic_class	Target: 0, 1, or 2

record_id is dropped because it is an identifier, not a feature.

## Workflow
Load the data and check for missing values.
Split into 80% training and 20% test (24 / 6 records), stratified by class.
Standardize the features, fitting the scaler on the training data only.
Train a one-layer model (5 inputs → 3 outputs) with cross-entropy loss and SGD (learning rate 0.03, batch size 8, 10 epochs).
Predict the class with the highest score and compute test accuracy.
Result

Test accuracy: 83.33% (5 of 6 test records classified correctly).

## How to run
bash
pip install pandas scikit-learn torch jupyter
jupyter notebook Class_assignment_CE_738.ipynb

Put transportation.csv in the same folder as the notebook.

## Limitations

With only 30 records, the 6-record test set gives a rough estimate: one misclassification changes accuracy by about 17 percentage points. Cross-validation would give a more reliable estimate.
