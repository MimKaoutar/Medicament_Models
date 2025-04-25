import os
import pandas as pd
import numpy as np
import json
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix,
                             f1_score, average_precision_score)
import torch
from typing import List, Tuple, Optional, Union


def load_dataset(
        train_file: str,
        dev_file: str,
        test_files: List[str],
        root_data: str = '/deepia/inutero/efemeris/data/efemeris_txt_v022025',
        variation: str = 'all',
        lang: str = 'en',
        encoder: bool = True
) -> Tuple[List[Union[pd.DataFrame, np.ndarray]],
           List[Union[pd.DataFrame, np.ndarray]],
           List[Union[pd.DataFrame, np.ndarray]],
           List[Union[pd.DataFrame, np.ndarray]]]:
    # Initialize lists
    X, Y, eval_X, eval_Y = [], [], [], []

    def process_file(file_path: str, split: str, is_eval: bool = False):
        """Helper function to process and load dataset files."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type based on extension
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.csv':
            df = pd.read_csv(file_path, sep=";")
            df["txt"] = df["txt"].astype(str)


        elif file_extension == '.json' or file_extension == '.jsonl':
            # Process JSONL file
            rows = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        json_obj = json.loads(line)

                        # Convert completion value from 'yes'/'no' to 1/0
                        completion = json_obj.get('completion', '')
                        if isinstance(completion, str):
                            if completion.lower().strip() == 'yes':
                                label = 1
                            elif completion.lower().strip() == 'no':
                                label = 0
                        else:
                            # If it's already a number, keep it
                            label = int(completion) if completion else 0
                        # Map 'prompt' to 'txt' and 'completion' to 'MALFO_MAJ'
                        rows.append({
                            'txt': json_obj.get('prompt', ''),
                            'MALFO_MAJ': label
                        })
            df = pd.DataFrame(rows)
            df["txt"] = df["txt"].astype(str)

        input_length = df["txt"].apply(lambda x: len([word for word in x.split() if word.strip()]))
        print(f"{split} input length -> Max: {input_length.max()} | Min: {input_length.min()} | Mean: {input_length.mean():.2f}")

        X_data = df[["txt"]] if encoder else df["txt"].values
        Y_data = df[["MALFO_MAJ"]] if encoder else df["MALFO_MAJ"].values

        # X_data = X_data.head(100)
        # Y_data = Y_data.head(100)

        (eval_X if is_eval else X).append(X_data)
        (eval_Y if is_eval else Y).append(Y_data)

    # Process training and development sets
    for split, file_name in zip(["train", "dev"], [train_file, dev_file]):
        file_path = os.path.join(root_data, variation, f"{lang}_random_{split}", file_name)
        process_file(file_path, split)

    # Process test sets (as evaluation)
    for test_file in test_files:
        file_path = os.path.join(root_data, variation, f"{lang}_random_test", test_file)
        process_file(file_path, "test", is_eval=True)

    return X, Y, eval_X, eval_Y


def compute_metrics(predictions, y_test):#, vectors=False):
    #if vectors:
    #    y_pred = (predictions >= 0.5).int()  # Binarize predictions at threshold 0.5
    #else:
    y_pred = (predictions >= 0.5).astype(int)  # Binarize predictions at threshold 0.5
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate AUC-ROC
    auc_roc = roc_auc_score(y_test, predictions)
    
    # Calculate AUC-PRC (Area under the Precision-Recall Curve)
    auc_prc = average_precision_score(y_test, predictions)

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print the evaluation metrics
    return accuracy, f1, auc_roc, auc_prc, conf_matrix
    # print(f"Accuracy: {accuracy:.4f}")


def check_device():
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")


def default_converter(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.float64, np.float32)):
        return float(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")
