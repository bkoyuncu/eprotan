import pathlib
import os
import numpy as np
import pandas as pd
import torch.utils.data.dataset as torch_dataset
from torch.utils.data import DataLoader
import json
from typing import Tuple, Iterator, Literal

from gluonts.dataset.repository import get_dataset
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

# extend this list if needed
SUPPORTED_DATASETS = {
    'electricity_nips', # hourly
    'solar_nips', # hourly
    'wiki2000_nips', # daily
} 
# electricity_nips has constant 0s in dim = 322, remove it from the dataset?

DATASET_TEST_DATES = {
    'electricity_nips': 7,
    'solar_nips': 7,
    'wiki2000_nips': 5
}

PREDICTION_LENGTHS = {
    'electricity_nips': 24,
    'solar_nips': 24,
    'wiki2000_nips': 30
}



def get_gluonts_multivar_dataset(
    dataset_name: str,
    dataset_path: pathlib.Path = pathlib.Path(os.getcwd()) / "data_gluonts",
    regenerate: bool = False,
    prediction_length: int = None,
    print_stats: bool = True,
) -> TrainDatasets:
    """Load a dataset from GluonTS repository.

    Args:
        dataset_name (str): Name of the dataset to load.
        dataset_path (pathlib.Path, optional): Path where to save the downloaded dataset. Defaults to pathlib.Path(os.getcwd())/"data_gluonts".
        regenerate (bool, optional): Whether to regenerate (newly download) the dataset. Defaults to False.
        prediction_length (int, optional): Prediction length of the dataset. Defaults to None. Is usually given by the dataset itself.
        print_stats (bool, optional): Whether to print the dataset statistics. Defaults to True.
        
    Raises:
        ValueError: If the dataset is not supported.

    Returns:
        TrainDatasets: GluonTS dataset.
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    dataset = get_dataset(dataset_name, path=dataset_path, regenerate=regenerate, prediction_length=prediction_length)
    
    if print_stats:
        print(f"=== Training dataset statistics: for '{dataset_name}'. ===")
        print(calculate_dataset_statistics(dataset.train))
        print(f"=== Test dataset statistics: for '{dataset_name}'. ===")
        print(calculate_dataset_statistics(dataset.test))
    
    # group the data to make it multivariate
    train_grouper = MultivariateGrouper(train_fill_rule = np.mean , test_fill_rule = lambda x: 0.0)
    num_test_dates = DATASET_TEST_DATES[dataset_name]
    test_grouper = MultivariateGrouper(num_test_dates=num_test_dates, train_fill_rule = np.mean, test_fill_rule = lambda x: 0.0)
    
    train_data = train_grouper(dataset.train)
    test_data = test_grouper(dataset.test)
    
    dataset = TrainDatasets(metadata=dataset.metadata, train=train_data, test=test_data)
    
    
    print(f"Dataset '{dataset_name}' loaded from GluonTS.")
    return dataset


class GluonTSDataset(torch_dataset.Dataset):
    def __init__(
        self,
        dataset_name: str,
        kind: Literal['train', 'val', 'test'] = 'train',
        dataset_path: pathlib.Path = pathlib.Path(os.getcwd()) / "data_gluonts",
        num_validation_dates: int = 0,
        regenerate: bool = False,
        prediction_length: int = None,
        history_length: int = None,
        window_offset: int = None,
        random_offset: bool = False,
    ):
        """
        Args:
            dataset_name (str): Name of the dataset to load.
            kind (Literal['train', 'val', 'test'], optional): Train, Validation or Test set. Defaults to 'train'.
            dataset_path (pathlib.Path, optional): Path where to save the downloaded dataset. Defaults to pathlib.Path(os.getcwd())/"data_gluonts".
            num_validation_dates (int, optional): Number of dates to use for validation. Defaults to 0. If 0 or None, no validation set is created.
            regenerate (bool, optional): Whether to regenerate (newly download & create) the dataset. Defaults to False.
            prediction_length (int, optional): Prediction length of the dataset. Defaults to None. Is usually given by the dataset itself, which is set if None.
            history_length (int, optional): Conditioning length when predictiing. If None, it is set to the prediction_length. Defaults to None.
            window_offset (int, optional): Offset for the windowing of the dataset when batching. If None, it is set to the prediction_length + history_length, such that no sample overlaps. Defaults to None.
                The number of samples in the dataset will be (T - history_len - pred_len) // window_offset, where T is the length of the full training sequence.
            random_offset (bool, optional): Whether to randomly offset the windowing when __getitem__ is called. If true, the window_offset is randomly chosen between 0 and window_offset-1. Defaults to False.
        """
        assert kind in ['train', 'val', 'test'], "Kind must be 'train', 'val' or 'test'."
        assert window_offset is None or window_offset > 0, "Window offset must be positive."
        if prediction_length is not None:
            assert PREDICTION_LENGTHS[dataset_name] == prediction_length, f"Prediction length does not match the expected prediction length ({prediction_length} != {PREDICTION_LENGTHS[dataset_name]})."
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.kind = kind
        create_val_set = True
        if num_validation_dates == 0 or num_validation_dates is None:
            create_val_set = False
            num_validation_dates = 0
        self.num_validation_dates = num_validation_dates
        self.prediction_length = prediction_length
        self.history_length = history_length if history_length is not None else prediction_length
        self.window_offset = window_offset
        self.random_offset = random_offset
        
        # sets prediction_length and history_length if not already set
        self._download_and_create_dataset(dataset_path=dataset_path, regenerate=regenerate, prediction_length=prediction_length, history_length=history_length, create_val_set=create_val_set)
        
        assert self.prediction_length is not None, "Prediction length must be set."
        assert self.history_length is not None, "History length must be set."
        
        if random_offset and self.kind != 'train':
            print("Warning: random_offset is ignored for the test and validation sets.")
        
        if window_offset is None and self.kind == 'train':
            self.window_offset = self.prediction_length + self.history_length
        elif window_offset is not None and self.kind == 'train':
            self.window_offset = min(window_offset, self.prediction_length + self.history_length)
        elif window_offset is not None and self.kind != 'train':
            print("Warning: window_offset is ignored for the test and validation sets.")
        
        self.data_scaler = GluonTSSequenceScaler()
        self.covariates_scaler = GluonTSSequenceScaler()
        
        self.data, self.covariates, self.lag_covariates = self._get_data(kind=kind) # [dim, T], [dim_c, T], [dim_c_lag, T] (train) | [test_size, dim, history_length + prediction_length], [test_size, dim_c, history_length + prediction_length] [test_size, dim_c_lag, history_length + prediction_length] (test)
        
        if self.kind != 'train':
            self.length = self.data.shape[0]
        elif self.kind == 'train' and self.random_offset:
            self.length = (self.data.shape[1] - self.history_length - self.prediction_length) // self.window_offset
        else:
            self.length = (self.data.shape[1] - self.history_length - self.prediction_length) // self.window_offset + 1
         
        # to ensure that the last sample covers the last part of the sequence
        self.final_offset = None
        if self.kind == 'train' and self.random_offset:
            self.final_offset = (self.data.shape[1] % self.window_offset) + 1
        elif self.kind == 'train' and not self.random_offset:
            self.final_offset = self.data.shape[1] % self.window_offset
    
    
    def __len__(self):
        return self.length
    
    
    def __getitem__(self, idx):
        """
        Select a sample from the dataset

        Args:
            index (int): row for selecting one sample

        Returns:
            x: sample data                                                          (T, dim_x)
            c: covariate data                                                       (T, dim_c)
            T0: horizon index for separating history and forecasting windows        (1)
            T: length of each sequence                                              (1)
        """
        T = self.history_length + self.prediction_length
        T = np.array([T])
        T0 = np.array([self.history_length])
        
        if self.kind != 'train':
            x, c, c_lag =  self.data[idx].T, self.covariates[idx].T, self.lag_covariates[idx].T # transpose to match our framework dimensions [T, dim], [T, dim_c] [T, dim_c_lag]
        
        else:
            offset = 0
            if self.random_offset:
                offset = np.random.randint(self.window_offset) # in [0, window_offset - 1]
            if idx == self.length - 1:
                offset += np.random.randint(self.final_offset + 1) # in [0, final_offset]
            new_idx = idx * self.window_offset + offset
            
            x = self.data[:, new_idx:new_idx + self.history_length + self.prediction_length].T # [T, dim]
            c = self.covariates[:, new_idx:new_idx + self.history_length + self.prediction_length].T # [T, dim_c]
            c_lag = self.lag_covariates[:, new_idx:new_idx + self.history_length + self.prediction_length].T # [T, dim_c_lag]
          
        c = np.concatenate([c, c_lag], axis=1) # [T, dim_c + dim_c_lag]  
        return x, c, T0.astype(np.int32), T.astype(np.int32)
    
    
    def create_pandas_evaluation_iterator(self, kind: Literal['train', 'val', 'test'] = 'test') -> Iterator[pd.DataFrame]:
        """
        Create an iterator for evaluation using gluonts.evaluation.MultivariateEvaluator.
        
        Args:
            kind (Literal['train', 'val', 'test']): Train, Validation or Test set. Defaults to 'test'.

        Returns:
            Iterator: Iterator for evaluation. Note that the data is not normalized.
            List[pd.Period]: List of start dates for the forecast windows in same order as the iterator.
        """       
        print(f"Creating pandas iterator for evaluation of '{self.dataset_name}'.")
        if kind == 'val':
            raise NotImplementedError("Validation set is not supported for evaluation via gluonts.evaluation.MultivariateEvaluator as it is not part of the main functionality.")
        dataset = get_gluonts_multivar_dataset(
            self.dataset_name,
            dataset_path = self.dataset_path,
            prediction_length = self.prediction_length,
            print_stats = False
        )
        test_data = dataset.train if kind == 'train' else dataset.test
        data_list = []
        start_dates = []
        for entry in test_data:
            origianl_length = entry["target"].shape[1]
            truncated_data = entry["target"][:, - (self.history_length + self.prediction_length):] # [dim, history_length + prediction_length]
            new_start = entry["start"] + origianl_length - truncated_data.shape[1]
            data_list.append(
                pd.DataFrame(
                    data=truncated_data.T, # [history_length + prediction_length, dim]
                    index=pd.date_range(start=new_start.to_timestamp(), periods=truncated_data.shape[1], freq=new_start.freqstr).to_period(),
                ) 
            )
            start_dates.append(new_start + self.history_length)
        return iter(data_list), start_dates
        
       
    def _get_data(self, kind: Literal['train', 'test', 'val']) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the data and covariates from the dataset and scale them.
        
        Args:
            kind (Literal['train', 'test', 'val']): Train, Test or Validation set.
        """
        loaded_data = np.load(self.dataset_path / self.dataset_name / f"{kind}_split.npy", allow_pickle=True).item()
        data = loaded_data["data"] # [dim, T] | [test_size, dim, history_length + prediction_length]
        covariates = loaded_data["covariates"] # [dim_c, T] | [test_size, dim_c, history_length + prediction_length]
        lag_covariates = loaded_data["lag_covariates"] # [dim_c_lag, T] | [test_size, dim_c_lag, history_length + prediction_length]
        
        # do not normalize the lag covariates
        if kind == 'train':
            data = self.data_scaler.fit_transform(data)
            covariates = self.covariates_scaler.fit_transform(covariates)
            print(f"""
                    training data after normalization:
                    data mean = {data.mean()}
                    data var = {data.var()}
                    covariates mean = {covariates.mean()}
                    covariates var = {covariates.var()}
                    """)
        else:
            loaded_data = np.load(self.dataset_path / self.dataset_name / "train_split.npy", allow_pickle=True).item()
            train_data = loaded_data["data"]
            train_covariates = loaded_data["covariates"]
            self.data_scaler.fit(train_data)
            self.covariates_scaler.fit(train_covariates)
            data = self.data_scaler(data.transpose(0, 2, 1)).transpose(0, 2, 1)
            covariates = self.covariates_scaler(covariates.transpose(0, 2, 1)).transpose(0, 2, 1)
            
            print(f"""
                    {kind} data after normalization:
                    data mean = {data.mean()}
                    data var = {data.var()}
                    covariates mean = {covariates.mean()}
                    covariates var = {covariates.var()}
                    """)
            
        return data.astype(np.float32), covariates.astype(np.float32), lag_covariates.astype(np.float32)
    
    
    def _check_exist_gluonts_dataset(self):
        if (os.path.exists(self.dataset_path / self.dataset_name)
            and os.path.exists(self.dataset_path / self.dataset_name / "metadata.json")):
            return True
        else:
            return False
        
    
    def _check_exist_dataset(self):
        if (os.path.exists(self.dataset_path / self.dataset_name)
            and os.path.exists(self.dataset_path / self.dataset_name / "train_split.npy")
            and os.path.exists(self.dataset_path / self.dataset_name / "test_split.npy")):
            return True
        else:
            return False
        

    def _create_covariates(self, length: int, start: pd.Period) -> np.ndarray:
        """Create covariates for the dataset.

        Args:
            length (int): Length of the sequence
            start (pd.Period): Start date of the sequence with the sample frequency.

        Returns:
            np.ndarray: [dim_c, length] Covariates for the sequence.
        """
        
        # we can consider to use sine and cosine functions for the time covariates
        if start.freqstr == 'H':
            covariates = [[timestamp.month, timestamp.day, timestamp.hour] for timestamp in pd.date_range(start=start.to_timestamp(), periods=length, freq='H')]
            covariates = np.array(covariates).T # [3, length]
        elif start.freqstr == 'D':
            covariates = [[timestamp.month, timestamp.day] for timestamp in pd.date_range(start=start.to_timestamp(), periods=length, freq='D')]
            covariates = np.array(covariates).T # [2, length]
        else: 
            raise NotImplementedError("Only hourly frequency is supported.")
        return covariates
    
    
    def _create_lag_covariates(self, length: int, start: pd.Period) -> np.ndarray:
        """Create lag covariates for the dataset.

        Args:
            length (int): Length of the sequence
            start (pd.Period): Start date of the sequence with the sample frequency.

        Returns:
            np.ndarray: [dim_c_lag, length] Covariates for the sequence.
        """
        if start.freqstr == 'H':
            lag_covariates = [[168, 24, 1] for timestamp in pd.date_range(start=start.to_timestamp(), periods=length, freq='H')]
            lag_covariates = np.array(lag_covariates).T # [3, length]
        elif start.freqstr == 'D':
            lag_covariates = [[7, 1] for timestamp in pd.date_range(start=start.to_timestamp(), periods=length, freq='D')]
            lag_covariates = np.array(lag_covariates).T # [2, length]
        else:
            raise NotImplementedError("Only hourly frequency is supported.")
        return lag_covariates
        
            
    def _download_and_create_dataset(
        self,
        dataset_path: pathlib.Path = pathlib.Path(os.getcwd()) / "data_gluonts",
        regenerate: bool = False,
        prediction_length: int = None,
        history_length: int = None,
        create_val_set: bool = False,
    ):
        """
        This function downloads a GluonTS dataset and creates a PyTorch dataset as numpy arrays from a GluonTS dataset, and stores it in the dataset folder.
        The dataset is stored in the dataset_path/dataset_name folder with train_split.npy and test_split.npy files.
        
        Args:
            dataset_path (pathlib.Path, optional): Path where to save the downloaded dataset. Defaults to pathlib.Path(os.getcwd())/"data_gluonts".
            regenerate (bool, optional): Whether to regenerate (newly download & create) the dataset. Defaults to False.
            prediction_length (int, optional): Prediction length of the dataset. Defaults to None. Is usually given by the dataset itself, which is set if None.
            history_length (int, optional): Conditioning length when predictiing. If None, it is set to the prediction_length. Defaults to None.
        """

        # need to download/load the dataset from GluonTS and create the dataset for PyTorch
        if regenerate or (not self._check_exist_dataset()):
            print(f"Creating dataset '{self.dataset_name}' from GluonTS repository.")
            dataset = get_gluonts_multivar_dataset(self.dataset_name, dataset_path=dataset_path, regenerate=regenerate, prediction_length=prediction_length)
            if prediction_length is None:
                self.prediction_length = dataset.metadata.prediction_length
            if history_length is None:
                self.history_length = self.prediction_length
            
            train_data = None
            train_covariates = None
            train_lag_covariates = None
            val_data = None if not create_val_set else []
            val_covariates = None if not create_val_set else []
            val_lag_covariates = None if not create_val_set else []
            test_data = []
            test_covariates = []
            test_lag_covariates = []
            
            for i, entry in enumerate(dataset.train):
                if i == 0:
                    data = entry["target"] # [dim, T]         
                    covariates = self._create_covariates(data.shape[1], entry["start"])
                    lag_covariates = self._create_lag_covariates(data.shape[1], entry["start"])
                    assert covariates.shape[1] == data.shape[1], "Covariates and data must have the same length."
                    assert lag_covariates.shape[1] == data.shape[1], "Lag covariates and data must have the same length."
                    train_data = data
                    train_covariates = covariates
                    train_lag_covariates = lag_covariates
                else:
                    raise ValueError("Expecting only one entry in the train dataset.")
                
            if create_val_set:
                # cut off num_validation_dates from the end of the training set
                len_train = train_data.shape[1]
                length_cutoff = self.num_validation_dates * self.prediction_length
                for num_val in range(self.num_validation_dates):
                    start_idx = len_train - length_cutoff + num_val * self.prediction_length - self.history_length
                    end_idx = len_train - length_cutoff + (num_val + 1) * self.prediction_length
                    val_data.append(train_data[:, start_idx:end_idx])
                    val_covariates.append(train_covariates[:, start_idx:end_idx])
                    val_lag_covariates.append(train_lag_covariates[:, start_idx:end_idx])
                    
                val_data = np.array(val_data) # [num_validation_dates, dim, history_length + prediction_length]
                val_covariates = np.array(val_covariates) # [num_validation_dates, dim_c, history_length + prediction_length]
                val_lag_covariates = np.array(val_lag_covariates) # [num_validation_dates, dim_c_lag, history_length + prediction_length]
                train_data = train_data[:, :-length_cutoff] # [dim, T - num_validation_dates * prediction_length]
                train_covariates = train_covariates[:, :-length_cutoff] # [dim_c, T - num_validation_dates * prediction_length]
                train_lag_covariates = train_lag_covariates[:, :-length_cutoff] # [dim_c_lag, T - num_validation_dates * prediction_length]
                    
                                       
            for entry in dataset.test:
                # only consider a sequence of length (history_length + prediction_length)
                original_length = entry["target"].shape[1]
                data = entry["target"][:, - (self.history_length + self.prediction_length):] # [dim, history_length + prediction_length]
                covariates = self._create_covariates(data.shape[1], entry["start"] + original_length - data.shape[1])
                lag_covariates = self._create_lag_covariates(data.shape[1], entry["start"] + original_length - data.shape[1])
                
                assert data.shape[1] == self.history_length + self.prediction_length, "Data must have the correct length."
                assert covariates.shape[1] == data.shape[1], "Covariates and data must have the same length."
                assert lag_covariates.shape[1] == data.shape[1], "Lag covariates and data must have the same length."
                
                test_data.append(data)
                test_covariates.append(covariates)
                test_lag_covariates.append(lag_covariates)
            
            test_data = np.array(test_data) # [test_size, dim, history_length + prediction_length]
            test_covariates = np.array(test_covariates) # [test_size, dim_c, history_length + prediction_length]
            test_lag_covariates = np.array(test_lag_covariates) # [test_size, dim_c_lag, history_length + prediction_length]
            
            # save the statistics in the dataset folder as json
            with open(dataset_path / self.dataset_name / "stats.json", "w") as f:
                json.dump({
                    "data_dim": train_data.shape[0],
                    "train_data_length": train_data.shape[1],
                    "covariates_dim": train_covariates.shape[0], 
                    "lag_covariates_dim": train_lag_covariates.shape[0],
                    "prediction_length": self.prediction_length, 
                    "history_length": self.history_length,
                    "num_test_dates": DATASET_TEST_DATES[self.dataset_name],
                    "num_validation_dates": self.num_validation_dates
                    }, f)
                
            print(f"Dataset '{self.dataset_name}' created and saved in '{dataset_path}'.")
            print(f"Train data shape: {train_data.shape}, Train covariates shape: {train_covariates.shape}, Train lag covariates shape: {train_lag_covariates.shape}.")
            if create_val_set:
                print(f"Validation data shape: {val_data.shape}, Validation covariates shape: {val_covariates.shape}, Validation lag covariates shape: {val_lag_covariates.shape}.")
            else:
                print("No validation set created.")
            print(f"Test data shape: {test_data.shape}, Test covariates shape: {test_covariates.shape}, Test lag covariates shape: {test_lag_covariates.shape}.")
                
            # save the dataset as numpy arrays
            train_data = {"data": train_data, "covariates": train_covariates, "lag_covariates": train_lag_covariates}
            test_data = {"data": test_data, "covariates": test_covariates, "lag_covariates": test_lag_covariates}
            np.save(dataset_path / self.dataset_name / "train_split.npy", train_data)
            np.save(dataset_path / self.dataset_name / "test_split.npy", test_data)
            
            if create_val_set:
                val_data = {"data": val_data, "covariates": val_covariates, "lag_covariates": val_lag_covariates}
                np.save(dataset_path / self.dataset_name / "val_split.npy", val_data)
            
        else:
            with open(dataset_path / self.dataset_name / "stats.json") as f:
                stats = json.load(f)
                assert stats["num_test_dates"] == DATASET_TEST_DATES[self.dataset_name], f"Number of test dates {stats['num_test_dates']} does not match the materialized dataset ({DATASET_TEST_DATES[self.dataset_name]}). Set regenerate=True to regenerate the dataset."
                assert stats["num_validation_dates"] == self.num_validation_dates, f"Number of validation dates {self.num_validation_dates} does not match the materialized dataset ({stats['num_validation_dates']}). Set regenerate=True to regenerate the dataset."
                if prediction_length is None:
                    self.prediction_length = stats["prediction_length"]
                else:
                    assert prediction_length == stats["prediction_length"], f"Prediction length does not match the materialized dataset ({prediction_length} != {stats['prediction_length']}). Set regenerate=True to regenerate the dataset."
                if history_length is None:
                    self.history_length = stats["history_length"]
                else:
                    assert history_length == stats["history_length"], f"History length does not match the materialized dataset ({history_length} != {stats['history_length']}). Set regenerate=True to regenerate the dataset."
                    
    
        
class ElectricityNIPS(GluonTSDataset):
    def __init__(
        self,
        kind: Literal['train', 'val', 'test'] = 'train',
        dataset_path: pathlib.Path = pathlib.Path(os.getcwd()) / "data_gluonts",
        num_validation_dates: int = 0,
        regenerate: bool = False,
        prediction_length: int = None,
        history_length: int = None,
        window_offset: int = None,
        random_offset: bool = False,
    ): 
        super(ElectricityNIPS, self).__init__(dataset_name="electricity_nips", kind=kind, dataset_path=dataset_path, num_validation_dates=num_validation_dates, 
                                              regenerate=regenerate, prediction_length=prediction_length, history_length=history_length,
                                              window_offset=window_offset, random_offset=random_offset)
       
    
class SolarNIPS(GluonTSDataset): 
    def __init__(
        self,
        kind: Literal['train', 'val', 'test'] = 'train',
        dataset_path: pathlib.Path = pathlib.Path(os.getcwd()) / "data_gluonts",
        num_validation_dates: int = 0,
        regenerate: bool = False,
        prediction_length: int = None,
        history_length: int = None,
        window_offset: int = None,
        random_offset: bool = False,
    ): 
        super(SolarNIPS, self).__init__(dataset_name="solar_nips", kind=kind, dataset_path=dataset_path, num_validation_dates=num_validation_dates,
                                        regenerate=regenerate, prediction_length=prediction_length, history_length=history_length,
                                        window_offset=window_offset, random_offset=random_offset)
        
        
class Wiki2000NIPS(GluonTSDataset):
    def __init__(
        self,
        kind: Literal['train', 'val', 'test'] = 'train',
        dataset_path: pathlib.Path = pathlib.Path(os.getcwd()) / "data_gluonts",
        num_validation_dates: int = 0,
        regenerate: bool = False,
        prediction_length: int = None,
        history_length: int = None,
        window_offset: int = None,
        random_offset: bool = False,
    ): 
        super(Wiki2000NIPS, self).__init__(dataset_name="wiki2000_nips", kind=kind, dataset_path=dataset_path, num_validation_dates=num_validation_dates,
                                           regenerate=regenerate, prediction_length=prediction_length, history_length=history_length,
                                           window_offset=window_offset, random_offset=random_offset)
    

class GluonTSSequenceScaler:
    """
    A Scaler class for sequences loaded from GluonTS datasets.

    """

    def __call__(self, sequences):
        return self.transform(sequences)

    def fit(self, data: np.array, eps=1e-7):
        """Save the means and stds of a training set.
        Note: Assumes that the data is already cleaned (does not contain NaNs or infs).
        
        Args:
            data (np.array): training set of sequences (N, dim, T) or (dim, T)
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-7.
        """
        if data.ndim == 2:
            sequences = [data]
        elif data.ndim == 3:
            sequences = [seq for seq in data]
        else:
            raise NotImplementedError("Data must be either 2D or 3D.")
        
        sequences = np.hstack(sequences) # [dim, T] | [dim, T * N]
        means = np.mean(sequences, axis=1) # [dim]
        stds = np.std(sequences, axis=1) # [dim]
        
        print(f"Sequences fit with avg. mean: {means.mean()} and avg. stds: {stds.mean()}. (Avg. is over dimensions)")
        self.means = means.reshape(1, -1) # [1, dim]
        self.stds = stds.reshape(1, -1) # [1, dim]
        self.eps = eps

    # note that we change T and dim to stay consistent with our framework
    def transform(self, sequences: np.ndarray):
        """Apply the normalization

        Args:
            sequences (np.ndarray): single sequence or batch of sequences  (T, dim) | (N, T, dim)

        Returns:
            (np.ndarray): normalized single sequence or batch of sequences   (T, dim) | (N, T, dim)
        """
        if sequences.ndim == 2:
            means, stds = self.means, self.stds
        elif sequences.ndim == 3:
            means, stds = np.expand_dims(self.means, axis=1), np.expand_dims(self.stds, axis=1)
        else:
            raise NotImplementedError("Data must be either 2D or 3D.")
        # broadcasts automatically
        return (sequences - means) / (stds + self.eps)
            
    def fit_transform(self, sequences):
        """Fit the scaler and transform in one step

        Args:
            sequences (np.array): train_set (dim, T) | (N, dim, T)

        Returns:
            (np.array): normalized batch of sequences (dim, T) | (N, dim, T) 
        """
        self.fit(sequences)
        if sequences.ndim == 2:
            return self.transform(sequences.T).T
        elif sequences.ndim == 3:
            return self.transform(sequences.transpose(0, 2, 1)).transpose(0, 2, 1)
        else:
            raise NotImplementedError("Data must be either 2D or 3D.")
        
    # again note that we change T and dim to stay consistent with our framework
    def inverse_transform(self, sequences_normalized):
        """Apply the DEnormalization

        Args:
            sequences_normalized (np.array): single sequence or batch of normalized sequences (T, dim) | (N, T, dim)

        Returns:
            (np.array): single sequence or batch of denormalized sequences (T, dim) | (N, T, dim)
        """
        if sequences_normalized.ndim == 2:
            means, stds = self.means, self.stds
        elif sequences_normalized.ndim == 3:
            means, stds = np.expand_dims(self.means, axis=1), np.expand_dims(self.stds, axis=1)
        else:
            raise NotImplementedError("Data must be either 2D or 3D.")
        return sequences_normalized * (stds + self.eps) + means

    

# ====================== Data Loaders ======================
def get_gluonts_data_loader(
    dataset_name: str, 
    split='train', 
    path: pathlib.Path = pathlib.Path(os.getcwd()) / "data_gluonts",
    num_validation_dates: int = 0,
    regenerate: bool = False,
    prediction_length: int = None,
    history_length: int = None,
    window_offset: int = None,
    random_offset: bool = False,
    batch_size=64, 
    num_workers=4, 
    shuffling= True, 
    persistent_workers = False, 
    **kwargs
) -> DataLoader:
    if dataset_name == "electricity_nips":
        data = ElectricityNIPS(kind=split, dataset_path=path, regenerate=regenerate, num_validation_dates=num_validation_dates,
                                  prediction_length=prediction_length, history_length=history_length, 
                                  window_offset=window_offset, random_offset=random_offset)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffling, num_workers=num_workers, persistent_workers=persistent_workers)
    elif dataset_name == "solar_nips":
        data = SolarNIPS(kind=split, dataset_path=path, regenerate=regenerate, num_validation_dates=num_validation_dates,
                         prediction_length=prediction_length, history_length=history_length, 
                         window_offset=window_offset, random_offset=random_offset)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffling, num_workers=num_workers, persistent_workers=persistent_workers)
    elif dataset_name == "wiki2000_nips":
        data = Wiki2000NIPS(kind=split, dataset_path=path, regenerate=regenerate, num_validation_dates=num_validation_dates,
                            prediction_length=prediction_length, history_length=history_length, 
                            window_offset=window_offset, random_offset=random_offset)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffling, num_workers=num_workers, persistent_workers=persistent_workers)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    return data_loader
