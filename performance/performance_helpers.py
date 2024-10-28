"""
Python script to compute the recall@10 metric used by our supervisor to test our implementations.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def recall_at_10(file_path: str):
    """Compute basically a True value if the wanted docids is in the top 10 retrieved docids"""
    # Start by loading the aggregate solution file
    solutions = pd.read_csv(file_path)

    # Now check if the retrieved docids are the one we wanted
    solutions['correct'] = solutions.apply(
        lambda row: row['positive_docs'] in row['docids'],
        axis=1
    )

    # Save the concatenated dataframe or return it
    solutions.to_csv(file_path, index=False)
    print(f"Process for recall@10 done !")


def bm25s_concatenate_output_and_solution(outputs_directory: str = 'dev_matches',
                                          solution_path: str = '../data/dev.csv'):
    """Take all the BM25s outputs and concatenate them with the solutions."""
    # First load the solutions file
    solutions = pd.read_csv(solution_path)[['query_id', 'positive_docs', 'lang']]

    # List all the output we have (make sure to only take the bm25s.csv files)
    files_name = [file for file in os.listdir(outputs_directory)
                  if file.endswith('.csv') and file.startswith('bm25s')]

    # Initialize an empty list to store dataframes
    outputs = []

    # Regex patterns for extracting 'process' and 'k1' values
    process_pattern = re.compile(r'bm25s_(lc(?:_sw(?:_l)?)?)_k1')
    k1_pattern = re.compile(r'k1_(\d+\.\d+)')

    # Load them one by one and concatenate them with the solutions from dev.csv
    for file in files_name:
        # Load the CSV file
        output = pd.read_csv(os.path.join(outputs_directory, file))

        # Extract the 'process' using regex
        process_match = process_pattern.search(file)
        process = process_match.group(1) if process_match else None

        # Extract the 'k1' value using regex
        k1_match = k1_pattern.search(file)
        k1 = float(k1_match.group(1)) if k1_match else None

        # Add the 'process' and 'k1' columns
        output['process'] = process
        output['k1'] = k1

        # Concatenate the output with the solutions
        output = pd.concat([solutions, output], axis=1)

        # Append the dataframe to the list
        outputs.append(output)

    # Optionally concatenate all dataframes into one if needed
    concatenated_df = pd.concat(outputs, axis=0, ignore_index=True)

    # Save the concatenated dataframe or return it
    concatenated_df.to_csv('bm25s_aggregate_solutions.csv', index=False)
    print(f"Processed files saved to 'bm25s_aggregate_solutions.csv'.")


def tf_idf_concatenate_output_and_solution(outputs_directory: str = 'dev_matches',
                                           solution_path: str = '../data/dev.csv'):
    """Take all the TFIDF outputs and concatenate them with the solutions."""
    # First load the solutions file
    solutions = pd.read_csv(solution_path)[['query_id', 'positive_docs', 'lang']]

    # List all the output we have (make sure to only take the bm25s.csv files)
    files_name = [file for file in os.listdir(outputs_directory)
                  if file.endswith('.csv') and file.startswith('tf_idf')]

    # Initialize an empty list to store dataframes
    outputs = []

    # Regex patterns for extracting 'process' and 'k1' values
    process_pattern = re.compile(r'tf_idf_(lc(?:_sw(?:_l)?)?)_dev')

    # Load them one by one and concatenate them with the solutions from dev.csv
    for file in files_name:
        # Load the CSV file
        output = pd.read_csv(os.path.join(outputs_directory, file))

        # Extract the 'process' using regex
        process_match = process_pattern.search(file)
        process = process_match.group(1) if process_match else None

        # Add the 'process' and 'k1' columns
        output['process'] = process

        # Concatenate the output with the solutions
        output = pd.concat([solutions, output], axis=1)

        # Append the dataframe to the list
        outputs.append(output)

    # Optionally concatenate all dataframes into one if needed
    concatenated_df = pd.concat(outputs, axis=0, ignore_index=True)

    # Save the concatenated dataframe or return it
    concatenated_df.to_csv('tf_idf_aggregate_solutions.csv', index=False)
    print(f"Processed files saved to 'tf_idf_aggregate_solutions.csv'.")


def k1_comparison(file_path: str = 'bm25s_aggregate_solutions.csv'):
    """TODO"""
    # Start by loading the aggregate solution file
    solutions = pd.read_csv(file_path)

    # Group by process and k1 values
    solutions = solutions[['process', 'k1', 'correct']].groupby(['process', 'k1']).mean()

    # Initialize the plot
    plt.figure(figsize=(8, 6))

    # Plot using seaborn lineplot for better styling
    sns.lineplot(data=solutions, x='k1', y='correct', hue='process', marker='o')

    # Add labels and title
    plt.xlabel('k1 Value')
    plt.ylabel('Correct Score')
    plt.title('Correct Score vs k1 for Different Processes')

    # Display the plot
    plt.show()


def bm25s_lang_comparison(file_path: str = 'bm25s_aggregate_solutions.csv'):
    """Generate bar plots comparing language performance from BM25s across processes."""

    # Start by loading the aggregate solution file
    solutions = pd.read_csv(file_path)[['process', 'k1', 'lang', 'correct']]

    # List of unique processes
    processes = solutions['process'].unique()

    # Set up the subplots: 3 rows, 1 column (for each process)
    fig, axes = plt.subplots(len(processes), 1, figsize=(9, 11), sharey=True)

    # Ensure axes is iterable even if only 1 subplot
    if len(processes) == 1:
        axes = [axes]

    # Loop over each process and create a subplot
    for i, process in enumerate(processes):
        # Filter the data for the current process
        process_data = solutions[solutions['process'] == process].groupby(['process', 'k1', 'lang']).mean()

        # Plot each subplot using seaborn
        sns.barplot(data=process_data, x='lang', y='correct', hue='k1', ax=axes[i])

        # Customize each subplot
        axes[i].set_title(f'Process: {process}')
        axes[i].set_xlabel('Language')
        axes[i].set_ylabel('Correct Score')
        axes[i].legend(title='k1 Value')
        axes[i].tick_params(axis='x', rotation=0)  # Rotate x-axis labels if needed

        # Add grid lines below the bars
        axes[i].grid(visible=True, linestyle='--', linewidth=0.7, alpha=0.7)

        # Set Y-axis limits to be between 0.6 and 1.0
        axes[i].set_ylim(0.4, 1.0)

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()


def tf_idf_lang_comparison(file_path: str = 'tf_idf_aggregate_solutions.csv'):
    """Generate a bar plot comparing language performance from TF-IDF across processes."""

    # Load the aggregate solution file
    solutions = pd.read_csv(file_path)[['process', 'lang', 'correct']]

    # Set up the figure and axis for a single plot
    fig, ax = plt.subplots(figsize=(9, 4))

    # Group by 'process' and 'lang' and calculate the mean 'correct' score
    process_data = solutions.groupby(['process', 'lang']).mean().reset_index()

    # Create a barplot with 'lang' on the x-axis, 'correct' on the y-axis, and 'process' as hue
    sns.barplot(data=process_data, x='lang', y='correct', hue='process', ax=ax)

    # Customize the plot
    ax.set_title('TF-IDF Language Performance Across Processes')
    ax.set_xlabel('Language')
    ax.set_ylabel('Correct Score')
    ax.tick_params(axis='x', rotation=0)  # Adjust x-axis labels if necessary

    # Add grid lines below the bars
    ax.grid(visible=True, linestyle='--', linewidth=0.7, alpha=0.7)

    # Set Y-axis limits to be between 0.6 and 1.0 (optional, adjust as necessary)
    ax.set_ylim(0.4, 1.0)

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def models_compare(bm25s_file_path: str = 'bm25s_aggregate_solutions.csv',
                   tf_idf_file_path: str = 'tf_idf_aggregate_solutions.csv',
                   k1_chosen: float = 1.2):
    """Generate a bar plot comparing BM25s and TF-IDF performance across processes."""

    # Load the aggregate solution files
    tf_idf_solutions = pd.read_csv(tf_idf_file_path)[['process', 'lang', 'correct']]
    bm25s_solutions = pd.read_csv(bm25s_file_path)[['process', 'lang', 'k1', 'correct']]

    # Only keep the BM25s model with the chosen k1
    bm25s_solutions = bm25s_solutions[bm25s_solutions['k1'] == k1_chosen].drop(columns=['k1'])

    # Add a 'model' column to distinguish between solutions
    tf_idf_solutions['model'] = 'tf_idf'
    bm25s_solutions['model'] = f'bm25s (k1: {k1_chosen})'

    # Concatenate both models' results
    solutions = pd.concat([tf_idf_solutions, bm25s_solutions])

    # Set up the figure and axis for a single plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create a barplot with 'lang' on the x-axis, 'correct' on the y-axis, and 'model' as hue
    sns.barplot(data=solutions, x='process', y='correct', hue='model', ax=ax, errorbar=('ci', 95))

    # Customize the plot
    ax.set_title('BM25s vs TF-IDF Performance Comparison by Process')
    ax.set_xlabel('Process')
    ax.set_ylabel('Correct Score')
    ax.tick_params(axis='x', rotation=0)

    # Add grid lines below the bars
    ax.grid(visible=True, linestyle='--', linewidth=0.7, alpha=0.7)

    # Set Y-axis limits to be between 0.4 and 1.0
    ax.set_ylim(0.4, 1.0)

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def models_compare_per_lang(bm25s_file_path: str = 'bm25s_aggregate_solutions.csv',
                            tf_idf_file_path: str = 'tf_idf_aggregate_solutions.csv',
                            k1_chosen: float = 1.2):
    """Generate bar plots comparing BM25s and TF-IDF performance for each process, grouped by language."""

    # Load the aggregate solution files
    tf_idf_solutions = pd.read_csv(tf_idf_file_path)[['process', 'lang', 'correct']]
    bm25s_solutions = pd.read_csv(bm25s_file_path)[['process', 'lang', 'k1', 'correct']]

    # Only keep the BM25s model with the chosen k1
    bm25s_solutions = bm25s_solutions[bm25s_solutions['k1'] == k1_chosen].drop(columns=['k1'])

    # Add a 'model' column to distinguish between solutions
    tf_idf_solutions['model'] = 'tf_idf'
    bm25s_solutions['model'] = f'bm25s (k1: {k1_chosen})'

    # Concatenate both models' results
    solutions = pd.concat([tf_idf_solutions, bm25s_solutions])

    # List of unique processes
    processes = solutions['process'].unique()

    # Set up subplots: one row per process, sharing the Y-axis
    fig, axes = plt.subplots(len(processes), 1, figsize=(9, 4 * len(processes)), sharey=True)

    # Ensure axes is iterable even if only one process exists
    if len(processes) == 1:
        axes = [axes]

    # Loop over each process and create a subplot
    for i, process in enumerate(processes):
        # Filter the data for the current process
        process_data = solutions[solutions['process'] == process]

        # Create a barplot comparing BM25s and TF-IDF for this process
        sns.barplot(data=process_data, x='lang', y='correct', hue='model', ax=axes[i], errorbar=('ci', 95))

        # Customize each subplot
        axes[i].set_title(f'Process: {process}')
        axes[i].set_xlabel('Language')
        axes[i].set_ylabel('Correct Score')
        axes[i].tick_params(axis='x', rotation=0)

        # Add grid lines below the bars
        axes[i].grid(visible=True, linestyle='--', linewidth=0.7, alpha=0.7)

        # Set Y-axis limits to be between 0.4 and 1.0
        axes[i].set_ylim(0.4, 1.0)

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()
