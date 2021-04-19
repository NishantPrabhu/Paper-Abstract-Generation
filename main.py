
""" 
Main script
"""

import argparse 
import trainers
from datetime import datetime as dt


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, type=str, help="Path to configuration file")
    ap.add_argument("-t", "--task", required=True, type=str, help="Task to perform, train or test")
    ap.add_argument("-l", "--load", required=False, type=str, help="Path to directory containing trained model")
    ap.add_argument("-o", "--output", required=False, default=dt.now().strftime("%H-%M_%d-%m-%Y"), help="Path to output dir")
    args = vars(ap.parse_args())

    trainer = trainers.DeepfakeClassifier(args)
    
    if args['task'] == 'train':
        trainer.train()

    elif args['task'] == 'test':
        assert args['load'] is not None, "Please load a model for generating test predictions"
        trainer.get_test_predictions()