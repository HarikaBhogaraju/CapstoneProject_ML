import os
import argparse
import sys
import setup
import train
import test

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='Test')
# parse the arguments
args = parser.parse_args()

def main():
	if(args.run == 'Train'):
		print("Setting up")
		setup.setData()
		print("Training")
		train.train_main()
	if(args.run == 'Test'):
		print("Setting up")
		setup.setData()
		print("Testing")
		test.test_main()

if __name__ == "__main__":
    main()
