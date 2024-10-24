"""
Helper to save results to submission.csv and compute it's recall@10.
Ps: Keep this file at the root of the project.
"""

import os
from typing import List, Union

import pandas as pd


def export_query_result_to_submission_csv(queryID: Union[str, List[str]],
                                          top_10_doc_ids: Union[List[str],
                                          List[List[str]]], mode: str = 'w') -> None:
    """
    Export the result of one or multiple queries to <project_root>/output/submission.csv.

    Args:
        queryID: The ID of the query or a list of query IDs.
        top_10_doc_ids: List of 10 retrieved doc IDs or a list of lists of 10 retrieved doc IDs.
        mode: 'w' to create a new file, 'a' to append to an existing file.
    """

    if (isinstance(queryID, str) and isinstance(top_10_doc_ids, list) and
            all(isinstance(doc_id, str) for doc_id in top_10_doc_ids)):
        # Single query case
        data = {'query_id': [queryID], 'docids': [top_10_doc_ids]}
    elif (isinstance(queryID, list) and all(isinstance(qid, str) for qid in queryID) and isinstance(top_10_doc_ids,
                                                                                                    list)
          and all(isinstance(doc_list, list) for doc_list in top_10_doc_ids)):
        # Multiple queries case
        data = {'query_id': queryID, 'docids': top_10_doc_ids}
    else:
        raise ValueError("Invalid input types for queryID and top_10_doc_ids")

    # Create a DataFrame with the queryID and the top 10 doc IDs
    df = pd.DataFrame(data)

    # Get the output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Export the DataFrame to a CSV file
    df.to_csv(os.path.join(output_dir, 'submission.csv'), mode=mode, header=(mode == 'w'), index=False)


def get_answers_for_submission_csv() -> None:
    """
    Create a new CSV file submission_answers.csv with an additional column stating if
    the retrieved top 10 documents for each query contain the relevant document.
    """

    # Get current directory
    current_dir = os.path.dirname(__file__)

    # Load the submission CSV file
    submission_df = pd.read_csv(os.path.join(current_dir, 'output', 'submission.csv'))

    train_df = pd.read_csv(os.path.join(current_dir, 'data', 'train.csv'))
    dev_df = pd.read_csv(os.path.join(current_dir, 'data', 'dev.csv'))

    # Merge the solution and submission DataFrames
    merged_df = pd.concat([train_df, dev_df]).merge(submission_df, on='query_id')

    # Check if the relevant document is in the top 10 retrieved documents
    merged_df['correct'] = merged_df.apply(lambda row: row['positive_docs'] in row['docids'], axis=1)
    merged_df = merged_df[['query_id', 'docids', 'correct']]

    # Export the merged DataFrame to a new CSV file
    merged_df.to_csv(os.path.join(current_dir, 'output', 'submission_answers.csv'), index=False)


def compute_recall_10() -> float:
    """
    Compute the recall@10 from the submission_answers.csv file.

    Returns:
        float: The recall@10 score.
    """

    # Get current directory
    current_dir = os.path.dirname(__file__)

    # Load the submission answers CSV file
    submission_answers_df = pd.read_csv(os.path.join(current_dir, 'output', 'submission_answers.csv'))

    # Compute the recall@10 score
    recall_10 = submission_answers_df['correct'].mean()

    return float(recall_10)
