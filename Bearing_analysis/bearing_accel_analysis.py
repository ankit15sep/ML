import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.axes import Axes
from typing import List, Dict
import glob, os
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

'''
Set No. 1:
    Recording Duration: October 22, 2003 12:06:24 to November 25, 2003 23:39:56
    No. of Files: 2,156
    No. of Channels: 8
    Channel Arrangement: Bearing 1 – Ch 1&2; Bearing 2 – Ch 3&4;
    Bearing 3 – Ch 5&6; Bearing 4 – Ch 7&8.
    File Recording Interval: Every 10 minutes (except the first 43 files were taken every 5 minutes)
    File Format: ASCII
    Description: At the end of the test-to-failure experiment, inner race defect occurred in
    bearing 3 and roller element defect in bearing 4.
Set No. 2:
    Recording Duration: February 12, 2004 10:32:39 to February 19, 2004 06:22:39
    No. of Files: 984
    No. of Channels: 4
    Channel Arrangement: Bearing 1 – Ch 1; Bearing2 – Ch 2; Bearing3 – Ch3; Bearing 4 – Ch 4.
    File Recording Interval: Every 10 minutes
    File Format: ASCII
    Description: At the end of the test-to-failure experiment, outer race failure occurred in
    bearing 1.
Set No. 3
    Recording Duration: March 4, 2004 09:27:46 to April 4, 2004 19:01:57
    No. of Files: 4,448
    No. of Channels: 4
    Channel Arrangement: Bearing1 – Ch 1; Bearing2 – Ch 2; Bearing3 – Ch3; Bearing4 – Ch4;
    File Recording Interval: Every 10 minutes
    File Format: ASCII
    Description: At the end of the test-to-failure experiment, outer race failure occurred in
    bearing 3.
'''

test1 = pd.read_csv("data/processed/1st_test_kpis.csv")
test1['date'] = pd.to_datetime(test1['date'])
test1 = test1.set_index('date').sort_index()

test2 = pd.read_csv("data/processed/2nd_test_kpis.csv")
test2['date'] = pd.to_datetime(test2['date'])
test2 = test2.set_index('date').sort_index()

test3 = pd.read_csv("data/processed/3rd_test_kpis.csv")
test3['date'] = pd.to_datetime(test3['date'])
test3 = test3.set_index('date').sort_index()


def get_psd(data: np.ndarray, fs: float = 1000.0, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Power Spectral Density (PSD) of the given data.

    Args:
        data (np.ndarray): Input signal data.
        fs (float): Sampling frequency in Hz.

    Returns:
        np.ndarray: PSD of the input data.
        :param nperseg: buffer length to use for the FFT.
    """
    f, pxx = signal.welch(data, fs=fs, window='hann', nperseg=nperseg)
    return f, pxx


def plot_bearing_analysis(df: pd.DataFrame, bearing_name: str=None, test_name: str=None, fig: Figure=None, ax: Axes=None):

    b = df[df["bearing"] == f'{bearing_name}']

    label = f'{test_name} - {bearing_name}'

    kpi_0_0 = "std"
    kpi_1_0 = "kurtosis"
    kpi_2_0 = "skewness"
    kpi_3_0 = "shape_factor"

    ax[0, 0].plot(b.index, b[kpi_0_0], label=label)
    ax[0, 0].set_ylabel("Std Dev (g)")
    ax[1, 0].plot(b.index, b[kpi_1_0], label=label)
    ax[1, 0].set_ylabel("Kurtosis")
    ax[2, 0].plot(b.index, b[kpi_2_0], label=label)
    ax[2, 0].set_ylabel("Skewness")
    ax[3, 0].plot(b.index, b[kpi_3_0], label=label)
    ax[3, 0].set_ylabel("Shape Factor")

    kpi_0_1 = "rms"
    kpi_1_1 = "max"
    kpi_2_1 = "min"
    kpi_3_1 = "mean"

    ax[0, 1].plot(b.index, b[kpi_0_1], label=label)
    ax[0, 1].set_ylabel("RMS (g)")
    ax[1, 1].plot(b.index, b[kpi_1_1], label=label)
    ax[1, 1].set_ylabel("Max (g)")
    ax[2, 1].plot(b.index, b[kpi_2_1], label=label)
    ax[2, 1].set_ylabel("Min (g)")
    ax[3, 1].plot(b.index, b[kpi_3_1], label=label)
    ax[3, 1].set_ylabel("Mean (g)")

    for a in ax.flatten():
        a.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        a.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        a.grid(True, alpha=0.3)
        a.legend(loc='best', fontsize='small')

    fig.autofmt_xdate()
    return fig, ax


def save_plots(bearing_names: List = ["bearing_0", "bearing_1", "bearing_2", "bearing_3"],
               test2names: Dict = {"Test 1": test1, "Test 2": test2, "Test 3": test3}):
    for test_name, df in test2names.items():
        print(f"Processing {test_name}...")
        for bearing_name in bearing_names:
            fig, ax = plt.subplots(4, 2, figsize=(12, 6), sharex=True)
            if test_name == "Test 1":
                fig, ax = plot_bearing_analysis(df=df, bearing_name=f'{bearing_name}_x', test_name=test_name, fig=fig,
                                                ax=ax)
                fig, ax = plot_bearing_analysis(df=df, bearing_name=f'{bearing_name}_y', test_name=test_name, fig=fig,
                                                ax=ax)
            else:
                fig, ax = plot_bearing_analysis(df=df, bearing_name=f'{bearing_name}_x', test_name=test_name, fig=fig,
                                                ax=ax)
            plt.xlabel("Date")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            print(f"Saving plot for {test_name} - {bearing_name}...")
            plt.savefig(f"data/processed/{test_name}_{bearing_name}.png")


def plot_waterfall(psd_list, freq_axis, timestamps=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Pre-transform Frequency and PSD to Log10
    # Use a small floor (e.g., 1e-9) to avoid log(0)
    log_freq = np.log10(freq_axis)

    for i in range(0, len(psd_list), 40):
        log_psd = np.log10(np.clip(psd_list[i], 1e-9, None))
        y_coord = i
        # color = plt.cm.plasma(i / len(psd_list))

        ax.plot(log_freq, [y_coord] * len(log_freq), log_psd, alpha=0.8)

    # 2. Custom Formatter to make the labels look like "10^x"
    log_formatter = FuncFormatter(lambda x, pos: f"$10^{{{int(x)}}}$")

    ax.xaxis.set_major_formatter(log_formatter)
    ax.zaxis.set_major_formatter(log_formatter)

    # 3. Set Labels and Clean up View
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Time / Snapshot')
    ax.set_zlabel('PSD')
    ax.set_title('Bearing Degradation Waterfall Plot (Fixed Log Scales)')

    # Adjust view angle for better visibility
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.show()


def build_dataset(df, burn_in=0.05, benchmark_point=0.3):

    df_for_pca = df[["kurtosis", "shape_factor", "rms"]]
    df_for_pca_normalized = StandardScaler().fit_transform(df_for_pca)
    pca = PCA(n_components=1)
    hi = pca.fit_transform(df_for_pca_normalized)
    coeff = np.corrcoef(hi[:, 0], np.arange(0, len(hi)))
    if coeff[0, 1] < 0:
        hi = -hi
    df["pca_health_indicator"] = hi
    burnin = int(burn_in*len(df['pca_health_indicator']))
    cutoff = int(benchmark_point*len(df['pca_health_indicator']))
    mean_metric = np.mean(df.iloc[burnin:burnin+cutoff]['pca_health_indicator'])
    std_metric = np.std(df.iloc[burnin:burnin+cutoff]['pca_health_indicator'])
    threshold = mean_metric + 3 * std_metric
    df["threshold"] = threshold
    is_above = df["pca_health_indicator"].rolling(window=50).mean() > threshold
    # df["knee_point"] = df[df['label'] == 1].index.min()
    df["label"] = np.where(is_above, 1, 0)
    df.iloc[:burnin, df.columns.get_loc("label")] = 0
    df["knee_point"] = df[df['label'] == 1].index.min()
    df["label"] = np.where(df.index >= df["knee_point"], 1, 0)
    # print(df.index.max())
    # print(df.index.max().timestamp())
    df["rul"] = (df.index.max() - df.index).total_seconds()/3600
    df.loc[df["label"] == 0, "rul"] = 50
    return df


def feature_extraction():
    first_test = pd.read_csv("data/processed/1st_test_kpis.csv")
    second_test = pd.read_csv("data/processed/2nd_test_kpis.csv")
    third_test = pd.read_csv("data/processed/3rd_test_kpis.csv")
    x_test1 = pd.DataFrame(columns=["date", "bearing", "kurtosis", "rms", "shape_factor", "pca_health_indicator", "threshold", "label"])
    x_test2 = pd.DataFrame(columns=["date", "bearing", "kurtosis", "rms", "shape_factor", "pca_health_indicator", "threshold", "label"])
    x_test3 = pd.DataFrame(columns=["date", "bearing", "kurtosis", "rms", "shape_factor", "pca_health_indicator", "threshold", "label"])
    x_test1[["date", "bearing", "kurtosis", "rms", "shape_factor"]] = first_test[["date", "bearing", "kurtosis", "rms", "shape_factor"]]
    x_test2[["date", "bearing", "kurtosis", "rms", "shape_factor"]] = second_test[["date", "bearing", "kurtosis", "rms", "shape_factor"]]
    x_test3[["date", "bearing", "kurtosis", "rms", "shape_factor"]] = third_test[["date", "bearing", "kurtosis", "rms", "shape_factor"]]
    x_test1['date'] = pd.to_datetime(x_test1['date'])
    x_test1 = x_test1.set_index('date').sort_index()
    x_test1 = x_test1.groupby("bearing", group_keys=False).apply(build_dataset)
    x_test2['date'] = pd.to_datetime(x_test2['date'])
    x_test2 = x_test2.set_index('date').sort_index()
    x_test2 = x_test2.groupby("bearing", group_keys=False).apply(build_dataset)
    x_test3['date'] = pd.to_datetime(x_test3['date'])
    x_test3 = x_test3.set_index('date').sort_index()
    x_test3 = x_test3.groupby("bearing", group_keys=False).apply(build_dataset)
    return x_test1, x_test2, x_test3


t1, t2, t3 = feature_extraction()
t1.to_csv("data/processed/1st_test_features.csv", index=True)
t2.to_csv("data/processed/2nd_test_features.csv", index=True)
t3.to_csv("data/processed/3rd_test_features.csv", index=True)
bearing = 'bearing_2_x'
test = t3
burn_in = 0.05
benchmark_point = 0.3
plt.plot(test[test["bearing"] == f"{bearing}"].index, test[test["bearing"] == f"{bearing}"]['pca_health_indicator'], label="PCA Health Indicator")
plt.axhline(test[test["bearing"] == f"{bearing}"]["threshold"][0], color='orange', linestyle='--', label='Threshold (Mean + 3*Std)')
if test[test["bearing"] == f"{bearing}"]["knee_point"][0] is not pd.NaT:
    plt.axvline(test[test["bearing"] == f"{bearing}"]["knee_point"][0], color='red', linestyle='--', label='Knee Point')
burnin = int(burn_in*len(test['pca_health_indicator']))
cutoff = int(benchmark_point*len(test['pca_health_indicator']))
plt.axvspan(test.index[burnin], test.index[burnin+cutoff], color='green', alpha=0.3, label='Benchmark Period')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()
#
plt.plot(test[test["bearing"] == f"{bearing}"].index, test[test["bearing"] == f"{bearing}"]['rul'], label="RUL")
plt.plot(test[test["bearing"] == f"{bearing}"].index, test[test["bearing"] == f"{bearing}"]['label'], label="label")
# if test[test["bearing"] == f"{bearing}"]["knee_point"][0] is not pd.NaT:
#     plt.axvline(test[test["bearing"] == f"{bearing}"]["knee_point"][0], color='red', linestyle='--', label='Knee Point')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()


# test_data_path = "data/raw/1st_test/1st_test"
# collect_all_files = glob.glob(os.path.join(test_data_path, "*"))
# freq_axis = []
# psd_list = []
# for file in collect_all_files:
#     file_dt = datetime.strptime(Path(file).stem, "%Y.%m.%d.%H.%M")
#     if file_dt.date() == datetime(2003, 11, 22).date():
#         df = pd.read_csv(file, header=None, sep='\t')
#         bearing_x = df.iloc[:, 0]
#         f, px = get_psd(data=bearing_x, fs=20000.0, nperseg=16384)
#         # freqs.append(f)
#         psd_list.append(px)





# plot_waterfall(psd_list, f)





# file = r"data\raw\1st_test\1st_test\2003.10.22.12.06.24"
# df = pd.read_csv(file, header=None, sep='\t')
# bearing_0x = df.iloc[:, 0]
# plt.hist(bearing_0x, bins=64, alpha=0.5, label='bearing_0_x')
# bearing_0y = df.iloc[:, 1]
# plt.hist(bearing_0y, bins=64, alpha=0.5, label='bearing_0_y')
# # bearing_2x = df.iloc[:, 4]
# # plt.hist(bearing_2x, bins=50, alpha=0.5, label='bearing_2_x')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# f, px = get_psd(data=bearing_x, fs=20000.0, nperseg=16384)
# psd_list.append(px)
# plt.plot(f, px, label="2003.11.25")
# plt.xscale('log')
# plt.yscale('log')

# file = collect_all_files[0]
# df = pd.read_csv(file, header=None, sep='\t')
# bearing_x = df.iloc[:, 4]
# f, px = get_psd(data=bearing_x, fs=20000.0, nperseg=16384)
# psd_list.append(px)
# plt.plot(f, px, label="2003.11.20")
# plt.xscale('log')
# plt.yscale('log')
# plt.grid(True, alpha=0.3)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Power Spectral Density")
# plt.title("PSD Comparison")
# plt.legend()
# plt.show()



# psd_list = [np.random.rand(64) for _ in range(10)]  # Example PSD data
# freq_axis = np.linspace(0, 100, 64)  # Example frequency axis
# # freq_axis = np.linspace(0, 100, 64)
# plot_waterfall(psd_list, f)