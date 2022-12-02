Mayo Clinic: Harnessing artificial intelligence for detection of pathogens in
drinking water within resource-constrained contexts

Using cheap apparatus and machine learning, this project aims to classify
water samples as clean or dirty based on the presence of pathogens. The following
code details the ML model and its functioning. The ML model reads water sample images 
collected from a microflip and identifies if the sample contains pathogens or not by
classifying it as Clean or Dirty.


1. python main.py --run Train:
This command will train and save a model

2. python main.py --run Test:
This command will use the saved model and run the test data from the datasets folder
