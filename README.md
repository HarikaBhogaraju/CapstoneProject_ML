Mayo Clinic: Harnessing artificial intelligence for detection of pathogens in
drinking water within resource-constrained contexts
Using cheap apparatus and machine learning, this project aims to classify
water samples as clean or dirty based on the presence of pathogens. The following
code details the ML model and its functioning.

1. make setup:
This command downloads all the necessary dependencies and sets up the datasets
from the datasets folder.

2. make train:
This command will train an XGBoost model with the datasets and save a trained model
as a .h5 file.

3. make test:
This command will use the saved model and run it on the test dataset.
