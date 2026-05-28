import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def load_bearing_data(file_path):
    """
    Load bearing data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing the bearing data.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded bearing data.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_data(df, burn_in_frac=0.05, add_bearing_id=False):
    columns = ["kurtosis", "rms", "shape_factor", "pca_health_indicator", "rul"]
    if add_bearing_id:
        columns.insert(0, "bearing")
    burn_in = len(df) * burn_in_frac
    data = df[columns]
    return data.iloc[int(burn_in):].reset_index(drop=True)


def get_test_train_data(df_list, shuffle=True):
    train_data = pd.concat(df_list[:2], ignore_index=True)
    if shuffle:
        train_data = train_data.sample(frac=1, random_state=1)
    x_train, y_train = train_data.drop(columns=["rul"]), train_data["rul"]
    test_data = df_list[2]
    x_test, y_test = test_data.drop(columns=["rul"]), test_data[["bearing", "rul"]]
    return x_train, y_train, x_test, y_test


t1 = create_data(load_bearing_data('data/processed/1st_test_features.csv'))
t2 = create_data(load_bearing_data('data/processed/2nd_test_features.csv'))
t3 = create_data(load_bearing_data('data/processed/3rd_test_features.csv'), add_bearing_id=True)
x_train, y_train, x_test, y_test = get_test_train_data([t1, t2, t3])

model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(x_train, y_train)
bearing = 'bearing_2_x'
x_test_bearing = x_test[x_test['bearing'] == bearing].drop(columns=['bearing'])
y_test_bearing = y_test[y_test['bearing'] == bearing]
y_pred = model.predict(x_test_bearing)
y_pred_series = pd.Series(y_pred, index=y_test_bearing.index)
plt.plot(y_test_bearing['rul'], label='Actual RUL')
plt.plot(y_pred_series, label='Predicted RUL')
plt.plot(y_pred_series.rolling(window=50).mean(), label='Predicted RUL (Smoothed)')
plt.xlabel('Sample Index')
plt.ylabel('RUL')
plt.title('Actual vs Predicted RUL')
plt.legend()
plt.show()




