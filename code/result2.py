
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import re
import pandas as pd
from typing import Tuple, Dict, Optional

# Let's define a function to read through the log file and extract a snippet of the data for round 100
def get_round_100_data(log_file_path: str, round_number: int) -> str:
    """
    Read through the log file and extract a snippet of the data for round 100.
    """

    with open(log_file_path, 'r') as file:
        log_content = file.readlines()

    # Search for the beginning of round 100
    round_str = f"Round {round_number}"
    start_agg = None
    end_agg = None
    start_ext = None
    end_ext = None
    switch = False
    for i, line in enumerate(log_content):
        if round_str in line:
            switch = True
        if switch:
            if 'Aggregation test\n' in line:
                start_agg = i + 1
            if 'external test' in line:
                end_agg = i
                start_ext = i + 1
        # last round
        end_ext = i

    return log_content[start_agg:end_agg], log_content[start_ext:end_ext]

def parse_round_data(log_data: list) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Parse the round data to extract the performance metrics for clients and aggregation.

    Parameters:
    log_data - A list of log file lines for a particular round.

    Returns:
    Two dictionaries with average performance metrics for clients and aggregation.
    """
    # Regex patterns for client and aggregation metrics
    agg_metrics_regex = r"Client\d+ Accuracy: (?P<test_accuracy>[\d.]+) Loss: (?P<test_loss>[\d.]+) ASR1: (?P<asr1>[\d.]+) ASR2: (?P<asr2>[\d.]+) Recall1: (?P<recall1>[\d.]+) Recall2: (?P<recall2>[\d.]+)"

    # Find all client matches
    agg_matches = re.findall(agg_metrics_regex, ''.join(log_data), re.DOTALL)

    # Convert matches to DataFrames
    agg_df = pd.DataFrame(agg_matches, columns=["test_accuracy", "test_loss", "asr1", "asr2", "recall1", "recall2"])

    # Convert all numeric columns to float and calculate averages
    agg_df = agg_df.astype(float).mean().to_dict()

    return agg_df

import os
import glob

# 로그 파일이 있는 기본 디렉토리
base_dir = './log/'

# 각 프로젝트 디렉토리를 저장할 리스트
project_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# 최신 로그 파일 경로와 프로젝트 이름을 저장할 리스트
latest_log_files_with_project = []

# 각 프로젝트 폴더를 순회
for project_dir in project_dirs:
    # 해당 폴더 내의 모든 .log 파일을 찾음
    log_files = glob.glob(os.path.join(project_dir, '*.log'))
    
    # 파일이 없으면 건너뜀
    if not log_files:
        continue
    
    # 파일들을 생성 시간에 따라 정렬
    log_files.sort(key=os.path.getmtime, reverse=True)
    
    # 가장 최신의 파일을 리스트에 추가
    latest_log_files_with_project.append((log_files[0], os.path.basename(project_dir)))

# 프로젝트 이름으로 정렬
latest_log_files_with_project.sort(key=lambda x: x[1])

# 정렬된 최신 로그 파일의 경로 리스트
latest_log_files_sorted = [file for file, project in latest_log_files_with_project]

# make result df
agg_result_df = pd.DataFrame(columns=['project',"test_accuracy", "test_loss", "asr1", "asr2", "recall1", "recall2"])
ext_result_df = pd.DataFrame(columns=['project',"test_accuracy", "test_loss", "asr1", "asr2", "recall1", "recall2"])

round_number = 100
for i, log_file_path in enumerate(latest_log_files_sorted):

    # Extracting a snippet of data for round 100
    agg, ext = get_round_100_data(log_file_path, round_number)

    # Now let's parse the extracted data for round 100 and calculate the averages
    agg_averages = parse_round_data(agg)
    ext_averages = parse_round_data(ext)

    # Add the results to the DataFrame
    agg_result_df.loc[i] = [latest_log_files_with_project[i][1]] + list(agg_averages.values())
    ext_result_df.loc[i] = [latest_log_files_with_project[i][1]] + list(ext_averages.values())

# Save the results to CSV files
agg_result_df.to_csv('agg_result.csv', index=False)
ext_result_df.to_csv('ext_result.csv', index=False)

# Function to parse project string and extract parameters
def parse_project_string(project_str):
    project_type = re.search(r"^[A-Z]+", project_str).group()
    m_stage_str = re.search(r"m(\d+)", project_str)
    m_stage = float(m_stage_str.group(1)) / 10 if m_stage_str else None
    ps_stage_str = re.search(r"ps([\d.]+)", project_str)
    ps_stage = float(ps_stage_str.group(1)) if ps_stage_str else None
    return project_type, m_stage, ps_stage

# Function to create a pivot table for heatmap plotting
def create_pivot_table(df, project_type, metric):
    filtered_df = df[df['project_type'] == project_type]
    pivot_table = filtered_df.pivot_table(
        index='poisoning_stage', 
        columns='malicious_client_stage', 
        values=metric,
        aggfunc=np.mean
    )
    return pivot_table

# Function to determine the color map and value range based on metric
def determine_cmap_vrange(metric):
    if "asr" in metric:
        # ASR metrics have values inverted, but colors not inverted
        return "YlGnBu", 1.0, 0.0
    elif "loss" in metric:
        # Loss metrics have a range from 0.5 to 1.5
        return "YlGnBu", 0.5, 1.5
    else:
        # Accuracy and recall metrics have a range from 0.0 to 1.0
        return "YlGnBu", 0.0, 1.0

# Function to plot heatmap and save to file
def plot_heatmap(pivot_table, metric, project_type, folder_path, dataframe_name):
    cmap, vmin, vmax = determine_cmap_vrange(metric)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(f'Heatmap of {metric} for {project_type} Project ({dataframe_name})')
    plt.xlabel('Malicious Client Stage')
    plt.ylabel('Poisoning Stage')
    heatmap_filename = f'{dataframe_name}_{metric}_{project_type}.png'
    plt.savefig(os.path.join(folder_path, heatmap_filename))
    plt.close() # Close the figure to avoid displaying it in the notebook

# Main function to load data, parse, and plot heatmaps
def main():
    # Load the data from CSV files
    agg_result_df = pd.read_csv('agg_result.csv')
    ext_result_df = pd.read_csv('ext_result.csv')

    # Parse the 'project' column to extract parameters
    agg_result_df[['project_type', 'malicious_client_stage', 'poisoning_stage']] = \
        agg_result_df['project'].apply(lambda x: pd.Series(parse_project_string(x)))

    ext_result_df[['project_type', 'malicious_client_stage', 'poisoning_stage']] = \
        ext_result_df['project'].apply(lambda x: pd.Series(parse_project_string(x)))

    # Create result directory
    result_dir = 'result2'
    os.makedirs(result_dir, exist_ok=True)

    # Metrics to plot
    metrics = ['test_accuracy', 'test_loss', 'asr1', 'asr2', 'recall1', 'recall2']

    # Dataframes to process
    dataframes = {
        'local': agg_result_df,
        'external': ext_result_df
    }

    # Generate heatmaps for each metric and project type, for each dataframe
    for df_name, df in dataframes.items():
        # Project types to plot
        project_types = df['project_type'].unique()
        for metric in metrics:
            for project_type in project_types:
                pivot_table = create_pivot_table(df, project_type, metric)
                plot_heatmap(pivot_table, metric, project_type, result_dir, df_name)

if __name__ == "__main__":
    main()
