import os
from typing import List, Tuple
import dotenv
import random
dotenv.load_dotenv()

class DatasetLoader():
    LABEL_COL = 'label'

    DATASET_BATCHES = None
    SEED = None
    BATCH_SIZE = None

    def __init__(self, batch_size: int, seed: int = 100):
        if DatasetLoader.DATASET_BATCHES is None:
            DatasetLoader.DATASET_BATCHES = self.load_datasets(batch_size)
        if seed != DatasetLoader.SEED or batch_size != DatasetLoader.BATCH_SIZE:
            DatasetLoader.SEED = seed
            DatasetLoader.BATCH_SIZE = batch_size
            random.seed(seed)
            DatasetLoader.DATASET_BATCHES = self.load_datasets(batch_size)
    
    def split_dataset(  self,
                        rows: List[Tuple[List[float], int]],
                        batch_size: int = None,
                        )-> List[Tuple[List[List[float]], List[int]]]:
        """
        This function splits the dataset into batches
        
        Args:
        rows(List[Tuple[List[float], int]]): The dataset rows
        batch_size(int): The batch size

        Returns:
        A list with the dataset batches, each batch contains a list with labels
        and a list with features
        """
        if batch_size is None:
            batch_size = len(rows)
        
        batches = []
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            features, labels = zip(*batch)
            batches.append((list(features), list(labels)))
        
        return batches

    def parse_row(  self,
                    row: str)-> Tuple[List[float], int]:
        """
        This function parses a row from the dataset

        Args:
        row(str): The row to parse

        Returns:
        A tuple with the label and the features
        - int: The label
        - List[float]: The features
        """
        row = row.strip().split(',')
        y = int(row[0])
        x = [float(val) for val in row[1:]]
        return x, y

    def load_datasets(  self,
                        batch_size:int=None,
                        )-> Tuple[List[Tuple[List[List[float]], List[int]]],List[Tuple[List[List[float]], List[int]]]]:
        """
        This function loads the train and test datasets 
        and splits them into batches

        Args:
        batch_size(int): The batch size for the datasets

        Returns:
        A tuple with train and test dataset batches
        - Tuple[List[float], int]: x (array) and y (int) category for the train dataset
        - Tuple[List[float], int]: x (array) and y (int) category for the test dataset 

        """
        train_path = os.getenv('TRAIN_DATASET_PATH')
        test_path = os.getenv('TEST_DATASET_PATH')
        assert os.path.exists(train_path), "The train dataset path does not exist"
        assert os.path.exists(test_path), "The test dataset path does not exist"
        with open(train_path, 'r') as f:
            raw_train_rows = f.readlines()[1:]
        with open(test_path, 'r') as f:
            raw_test_rows = f.readlines()[1:]

        # Shuffle the datasets
        random.shuffle(raw_train_rows)
        random.shuffle(raw_test_rows)

        train_data:List[Tuple[List[float], int]] = [self.parse_row(row) for row in raw_train_rows]
        test_data:List[Tuple[List[float], int]] = [self.parse_row(row) for row in raw_test_rows]

        train_batches = self.split_dataset(train_data, batch_size)
        test_batches = self.split_dataset(test_data, batch_size)

        return train_batches, test_batches
    
    def get_datasets(   self)-> Tuple[List[Tuple[List[List[float]], List[int]]],List[Tuple[List[List[float]], List[int]]]]:
        """
        This function returns the train and test datasets

        Returns:
        A tuple with train and test datasets, each has:
        List[Tuple[List[List[float]], List[int]]]: n batches, each batch is a tuple
        with a list of features and a list of labels
        """
        return self.DATASET_BATCHES