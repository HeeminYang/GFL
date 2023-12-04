
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

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

# Function to plot heatmap and save to file
def plot_heatmap(pivot_table, metric, project_type, folder_path, dataframe_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.0, vmax=1.0)
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
        'agg': agg_result_df,
        'ext': ext_result_df
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
