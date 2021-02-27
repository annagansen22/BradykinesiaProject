import pandas as pd
from tqdm import tqdm

from classifier import Classifier


class Evaluator:
    """Evaluate a classifier on the dataset."""


    def __init__(self, clf, data, n_runs=30):
        """
        Initialize Evaluator instance.

        Args:
            clf    = [Classifier] classifier instance
            data   = [Key2PD] dataset class instance
            n_runs = [int] number of evaluation runs (default=30)
        """
        self.clf = clf
        self.data = data
        self.n_runs = n_runs
        self.scores = []

    def evaluate(self):
        """Evaluate the classifier."""
        for i in tqdm(range(self.n_runs)):         
            X_train, X_test, y_train, y_test = self.data.split(random_state=i)
            BCA_train, BCA_test, mAUC_train, mAUC_test = self.clf.fit_predict(X_train, 
                                                                              y_train, 
                                                                              X_test, 
                                                                              y_test)
            self.scores.append({ 
                                'BCA_train': BCA_train, 
                                'BCA_test': BCA_test, 
                                'mAUC_train': mAUC_train, 
                                'mAUC_test': mAUC_test 
                               })


    def get_scores(self):
        """
        Get the scores as a pandas dataframe.

        Returns [pd.DataFrame]:
            Dataframe of the scores.
        """
        return pd.DataFrame(self.scores)


    def export_to_csv(self, filename):    
        """
        Export the scores to a CSV file.

        Args:
            filename = [string] filename including path
        """     
        df = self.get_scores()
        df.to_csv(filename, index=False)
