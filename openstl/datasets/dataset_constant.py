dataset_parameters = {
    'wind': {
        'in_shape': [144, 2],
        'pre_seq_length': 144,
        'aft_seq_length': 144,
        'total_length': 288,
        'data_name': 'wind',
        'metrics': ['#time_series', 'mse', 'rmse', 'mae', 'mape']
    },
    'wind_kdd': {
        'in_shape': [144, 2],
        'pre_seq_length': 144,
        'aft_seq_length': 144,
        'total_length': 288,
        'data_name': 'wind',
        'metrics': ['#time_series', 'mse', 'rmse', 'mae', 'mape']
    }
}