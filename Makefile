setup:
	pip install -r requirements.txt

train:
	python setup.py
	python train.py

test:
	python setup.py
	python test.py
