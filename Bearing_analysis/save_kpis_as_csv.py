import numpy as np
import pandas as pd
import os
import glob
from typing import List, Dict
import tqdm
from scipy import signal, stats
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_bearing_data(file_path: str) -> pd.DataFrame:
    """
    Load bearing data from a file.

    Args:
        file_path (str): Path to the bearing data file.

    Returns:
        np.ndarray: Loaded bearing data.
    """
    return pd.read_csv(file_path, header=None, sep='\t')


class TimeSeriesAnalyzer:
    def __init__(self, data: np.ndarray = None, axis: int = 0):
        self.data = data
        self.axis = axis

    def rms(self):
        return np.sqrt(np.mean(self.data**2, axis=self.axis))

    def kurtosis(self):
        return stats.kurtosis(self.data, axis=self.axis)

    def skewness(self):
        return stats.skew(self.data, axis=self.axis)

    def shape_factor(self):
        return self.rms() / np.mean(np.abs(self.data), axis=self.axis)

    def maxval(self):
        return np.max(self.data, axis=self.axis)

    def minval(self):
        return np.min(self.data, axis=self.axis)

    def std(self):
        return np.std(self.data, axis=self.axis)

    def meanval(self):
        return np.mean(self.data, axis=self.axis)


def load_test_data(test_data_path: str) -> Dict[str, List[np.ndarray]]:
    """
    Load test data from the specified path.
    :param test_data_path:
    :return:
    """
    collect_all_files = glob.glob(os.path.join(test_data_path, "*"))
    kpis = defaultdict(list)

    if not collect_all_files:
        raise FileNotFoundError("No files found in the specified directory.")
    else:
        print(f"Found {len(collect_all_files)} files in the directory.")
        with tqdm.tqdm(total=len(collect_all_files), desc="Loading files") as pbar:
            for file in collect_all_files:
                df = load_bearing_data(file)
                data = df.to_numpy()
                tsa = TimeSeriesAnalyzer(data, axis=0)
                date = datetime.strptime(Path(file).stem, "%Y.%m.%d.%H.%M")
                rms = tsa.rms()
                kurtosis = tsa.kurtosis()
                skewness = tsa.skewness()
                shape_factor = tsa.shape_factor()
                maxval = tsa.maxval()
                minval = tsa.minval()
                stdval = tsa.std()
                meanval = tsa.meanval()
                kpis["date"].extend([date] * (data.shape[1]))
                if "1st_test" in test_data_path:
                    for id, i in enumerate(np.arange(0, data.shape[1], 2)):
                        kpis["bearing"].append(f"bearing_{id}_x")
                        kpis["rms"].append(rms[i])
                        kpis["kurtosis"].append(kurtosis[i])
                        kpis["skewness"].append(skewness[i])
                        kpis["shape_factor"].append(shape_factor[i])
                        kpis["max"].append(maxval[i])
                        kpis["min"].append(minval[i])
                        kpis["std"].append(stdval[i])
                        kpis["mean"].append(meanval[i])

                        kpis["bearing"].append(f"bearing_{id}_y")
                        kpis["rms"].append(rms[i+1])
                        kpis["kurtosis"].append(kurtosis[i+1])
                        kpis["skewness"].append(skewness[i+1])
                        kpis["shape_factor"].append(shape_factor[i+1])
                        kpis["max"].append(maxval[i+1])
                        kpis["min"].append(minval[i+1])
                        kpis["std"].append(stdval[i+1])
                        kpis["mean"].append(meanval[i+1])
                else:
                    for i in range(int(data.shape[1])):
                        kpis["bearing"].append(f"bearing_{i}_x")
                        kpis["rms"].append(rms[i])
                        kpis["kurtosis"].append(kurtosis[i])
                        kpis["skewness"].append(skewness[i])
                        kpis["shape_factor"].append(shape_factor[i])
                        kpis["max"].append(maxval[i])
                        kpis["min"].append(minval[i])
                        kpis["std"].append(stdval[i])
                        kpis["mean"].append(meanval[i])

                pbar.update(1)
    df_kpis = pd.DataFrame(kpis)
    # kpis["date"] = kpis["date"][0]
    # df_kpis = kpis
    return df_kpis


def resample_data(data: pd.DataFrame, source_feq: int = 20000, target_freq: int = 1000) -> pd.DataFrame:
    """
    Resample the data to a target frequency.
    """
    resampled_data = defaultdict(list)
    down_sample_factor = source_feq/target_freq
    for col in data.columns:
        if col != 'date':
            resampled_data[col].extend(signal.resample(data[col].values, int(len(data[col].values)/down_sample_factor)))
    resampled_data['date'] = data['date'][::int(down_sample_factor)].values.tolist()
    return pd.DataFrame(resampled_data)


def main():
    df_test1 = load_test_data('data/raw/1st_test/1st_test')
    df_test1.to_csv('data/processed/1st_test_kpis.csv', index=False)
    df_test2 = load_test_data('data/raw/2nd_test/2nd_test')
    df_test2.to_csv('data/processed/2nd_test_kpis.csv', index=False)
    df_test3 = load_test_data('data/raw/3rd_test/3rd_test')
    df_test3.to_csv('data/processed/3rd_test_kpis.csv', index=False)


if __name__ == "__main__":
    main()


# This script loads bearing data from specified directories and saves them as HDF5 files.

