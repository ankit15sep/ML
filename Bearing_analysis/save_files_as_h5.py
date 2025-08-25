import numpy as np
import pandas as pd
import os
import glob
from typing import List, Dict
import tqdm
from scipy import signal
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


def load_test_data(test_data_path: str) -> Dict[str, List[np.ndarray]]:
    """
    Load test data from the specified path.
    :param test_data_path:
    :return:
    """
    collect_all_files = glob.glob(os.path.join(test_data_path, "*"))
    collect_accel_file = defaultdict(list)

    if not collect_all_files:
        raise FileNotFoundError("No files found in the specified directory.")
    else:
        print(f"Found {len(collect_all_files)} files in the directory.")
        with tqdm.tqdm(total=len(collect_all_files), desc="Loading files") as pbar:
            for file in collect_all_files:
                df = load_bearing_data(file)
                date = datetime.strptime(Path(file).stem, "%Y.%m.%d.%H.%M")
                if "1st_test" in test_data_path:
                    collect_accel_file["date"].extend([date]*df.shape[0])
                    collect_accel_file["bearing_1_x"].extend(df[0].values.tolist())
                    collect_accel_file["bearing_1_y"].extend(df[1].values.tolist())
                    collect_accel_file["bearing_2_x"].extend(df[2].values.tolist())
                    collect_accel_file["bearing_2_y"].extend(df[3].values.tolist())
                    collect_accel_file["bearing_3_x"].extend(df[4].values.tolist())
                    collect_accel_file["bearing_3_y"].extend(df[5].values.tolist())
                    collect_accel_file["bearing_4_x"].extend(df[6].values.tolist())
                    collect_accel_file["bearing_4_y"].extend(df[7].values.tolist())
                else:
                    collect_accel_file["date"].extend([date]*df.shape[0])
                    collect_accel_file["bearing_1_x"].extend((df[0].values.tolist()))
                    collect_accel_file["bearing_2_x"].extend((df[0].values.tolist()))
                    collect_accel_file["bearing_3_x"].extend((df[0].values.tolist()))
                    collect_accel_file["bearing_4_x"].extend((df[0].values.tolist()))
                pbar.update(1)
    return collect_accel_file


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
    # test_data_1 = load_test_data('data/raw/1st_test/1st_test')
    # df_1st_test = pd.DataFrame(test_data_1)
    # df_1st_test.to_hdf('data/raw/combined_test1.h5', key='df', mode='w')
    # df_1st_test_resampled = resample_data(df_1st_test)
    # df_1st_test_resampled.to_hdf('data/raw/combined_test1_resample_1000Hz.h5', key='df', mode='w')
    #
    # test_data_2 = load_test_data('data/raw/2nd_test/2nd_test')
    # df_2nd_test = pd.DataFrame(test_data_2)
    # df_2nd_test.to_hdf('data/raw/combined_test2.h5', key='df', mode='w')
    # df_2nd_test_resampled = resample_data(df_2nd_test)
    # df_2nd_test_resampled.to_hdf('data/raw/combined_test2_resample_1000Hz.h5', key='df', mode='w')

    test_data_3 = load_test_data('data/raw/3rd_test/3rd_test')
    df_3rd_test = pd.DataFrame(test_data_3)
    df_3rd_test.to_hdf('data/raw/combined_test3.h5', key='df', mode='w')
    df_3rd_test_resampled = resample_data(df_3rd_test)
    df_3rd_test_resampled.to_hdf('data/raw/combined_test3_resample_1000Hz.h5', key='df', mode='w')


if __name__ == "__main__":
    main()


# This script loads bearing data from specified directories and saves them as HDF5 files.

