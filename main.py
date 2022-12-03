import os
import argparse
import sys
import setup
import train
import test

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='Test')
parser.add_argument('--trainingdata', type=str, default="./datasets/TrainingDataset/")
parser.add_argument('--testingdata', type=str, default="./datasets/TestingDataset/")
# parse the arguments
args = parser.parse_args()

def main():
	if(args.run == 'Train'):
		print("Training")
		pathData = args.trainingdata
		train.train_main(pathData)

	if(args.run == 'Test'):
		print("Testing")
		pathData = args.testingdata
		test.test_main(pathData)

if __name__ == "__main__":
    main()
