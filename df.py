import pandas as pd

# Creating a DataFrame for the Aggregation Test data
data_aggregation = {
    'Client': ['Client1', 'Client2', 'Client3', 'Client4', 'Client5',
               'Client6', 'Client7', 'Client8', 'Client9', 'Client10'],
    'Accuracy': [0.9257, 0.9172, 0.9264, 0.9143, 0.9241,
                 0.9358, 0.9301, 0.9216, 0.9134, 0.9185],
    'Loss': [0.1765, 0.2013, 0.1702, 0.2078, 0.1794,
             0.1574, 0.1635, 0.1843, 0.2129, 0.1998],
    'ASR1': [0.0429, 0.0387, 0.0416, 0.0391, 0.0435,
             0.0378, 0.0402, 0.0420, 0.0394, 0.0383],
    'ASR2': [0.0123, 0.0115, 0.0121, 0.0119, 0.0126,
             0.0108, 0.0114, 0.0125, 0.0117, 0.0116],
    'Recall1': [0.9267, 0.9181, 0.9273, 0.9150, 0.9249,
                0.9365, 0.9310, 0.9223, 0.9141, 0.9192],
    'Recall2': [0.9246, 0.9163, 0.9255, 0.9132, 0.9230,
                0.9347, 0.9289, 0.9205, 0.9123, 0.9176]
}

df_aggregation = pd.DataFrame(data_aggregation)

# Calculating the average performance of clients on the Aggregation test for each metric
average_performance_aggregation = df_aggregation.mean()

# Creating a DataFrame for the External Test data
data_external = {
    'Client': ['Client1', 'Client2', 'Client3', 'Client4', 'Client5',
               'Client6', 'Client7', 'Client8', 'Client9', 'Client10'],
    'Accuracy': [0.7356, 0.7221, 0.7398, 0.7253, 0.7337,
                 0.7462, 0.7405, 0.7318, 0.7237, 0.7289],
    'Loss': [0.5263, 0.5417, 0.5205, 0.5382, 0.5289,
             0.5114, 0.5177, 0.5302, 0.5424, 0.5359],
    'ASR1': [0.2641, 0.2589, 0.2628, 0.2603, 0.2647,
             0.2580, 0.2604, 0.2622, 0.2596, 0.2585],
    'ASR2': [0.0782, 0.0765, 0.0779, 0.0777, 0.0784,
             0.0760, 0.0766, 0.0783, 0.0775, 0.0774],
    'Recall1': [0.7365, 0.7230, 0.7407, 0.7262, 0.7346,
                0.7471, 0.7414, 0.7327, 0.7246, 0.7298],
    'Recall2': [0.7345, 0.7212, 0.7389, 0.7244, 0.7328,
                0.7453, 0.7396, 0.7309, 0.7228, 0.7282]
}

df_external = pd.DataFrame(data_external)

# Calculating the average performance of clients on the External test for each metric
average_performance_external = df_external.mean()

# Combining the averages into a single DataFrame for comparison
average_performance = pd.DataFrame({
    'Metric': average_performance_aggregation.index,
    'Aggregation Test Average': average_performance_aggregation.values,
    'External Test Average': average_performance_external.values
})

print('Average Performance of Clients on the Aggregation Test and External Test')