import argparse
import os
import json
import numpy as np
from tabulate import tabulate
from collections import defaultdict
import argparse


def format_value(value, significant_digits=4):
    formatted_nvalue = f"{value*100:.6f}"  # convert to percentage
    # len of formatted number without dot
    len_value = len(formatted_nvalue.replace('.', ''))
    if len_value > significant_digits: 
        formatted_number = formatted_nvalue[:significant_digits+1]
    else:
        formatted_number = formatted_nvalue
    return formatted_number


def get_table_data_numeric_from_table(table_data):
    """
    Extract numeric values from table_data, excluding dataset names, separator rows, and average rows.
    
    Args:
        table_data: List of rows, where each row is [dataset_name, value1, value2, ...]
    
    Returns:
        List of lists containing numeric values only (excluding first column and special rows).
        Values are converted from percentage strings to floats (percentage form, e.g., 45.67 for 45.67%).
    """
    table_data_numeric = []
    
    for row in table_data:
        # Skip separator rows and average rows
        if row[0] == "-----" or row[0] == "AVG":
            continue
        
        # Extract numeric values (skip first column which is dataset name)
        row_values = []
        for val in row[1:]:  # Skip first column (dataset name)
            try:
                # Convert string percentage to float
                numeric_val = float(val)
                row_values.append(numeric_val)
            except (ValueError, TypeError):
                # If conversion fails, use None or NaN
                row_values.append(None)
        
        table_data_numeric.append(row_values)

    return np.array(table_data_numeric)
    # table_data_numeric = np.array(table_data_numeric)
    # table_data_numeric = np.row_stack([table_data_numeric, np.mean(table_data_numeric, axis=0)])
    # return table_data_numeric


def get_difference_matrix(table_data_numeric):
    """
    Compute difference matrix from table_data_numeric array.
    
    For each even column index i in {2, 4, 6, 8}, computes:
    - col_idx_i - col_idx_0 (base difference)
    - col_idx_{i+1} - col_idx_1 (novel difference)
    
    Args:
        table_data_numeric: numpy array of shape (n_datasets, n_columns)
                       where columns 0,1 are zeroshot (base, novel)
                       and columns 2,3,4,5,... are other methods (base, novel pairs)
    
    Returns:
        numpy array of shape (n_datasets, n_diff_columns) containing differences
    """
    table_data_numeric = np.array(table_data_numeric)
    n_rows, n_cols = table_data_numeric.shape
    
    # Collect difference columns
    diff_columns = []
    
    # Automatically find all even indices starting from 2 (base columns of each method)
    # Columns 0,1 are zeroshot, so we start from 2
    # Each method has 2 columns (base, novel), so we iterate by 2
    for i in range(2, n_cols, 2):
        if (i + 1) < n_cols:
            # Compute col_idx_i - col_idx_0 (base difference)
            base_diff = table_data_numeric[:, i] - table_data_numeric[:, 0]
            diff_columns.append(base_diff)
            
            # Compute col_idx_{i+1} - col_idx_1 (novel difference)
            novel_diff = table_data_numeric[:, i + 1] - table_data_numeric[:, 1]
            diff_columns.append(novel_diff)
    
    # Stack columns horizontally to form difference matrix
    if diff_columns:
        difference_matrix = np.column_stack(diff_columns)
    else:
        difference_matrix = np.array([]).reshape(n_rows, 0)
    
    return difference_matrix


def difference_matrix_to_strings(difference_matrix, significant_digits=3):
    r"""
    Convert difference matrix to list of strings with up/down triangle notation.
    
    Args:
        difference_matrix: numpy array of shape (n_datasets, n_diff_columns) 
                          containing differences in percentage form (e.g., 5.0 for 5.0%)
        significant_digits: Number of significant digits for formatting (default: 3)
    
    Returns:
        List of lists of strings, where each string is formatted as:
        - \uptri{value} if value >= 0
        - \downtri{value} if value < 0
    """
    difference_matrix = np.array(difference_matrix)
    n_rows, n_cols = difference_matrix.shape
    
    result_strings = []
    
    for row_idx in range(n_rows):
        row_strings = []
        for col_idx in range(n_cols):
            value = difference_matrix[row_idx, col_idx]
            
            # Handle NaN or None values
            if np.isnan(value) or value is None:
                row_strings.append("NA")
                continue
            
            # Convert from percentage to [0, 1] range for format_value
            value_normalized = value / 100.0
            
            # Format the value
            formatted_value = format_value(abs(value_normalized), significant_digits=significant_digits)
            
            # Add up or down triangle based on sign
            if value >= 0:
                row_strings.append(f"\\uptri{{{formatted_value}}}")
            else:
                row_strings.append(f"\\downtri{{{formatted_value}}}")
        
        result_strings.append(row_strings)
    
    return result_strings


def difference_matrix_to_plusminus_strings(difference_matrix, significant_digits=3):
    r"""
    Convert difference matrix to list of strings using +(...) / -(...) notation.

    Returns a list of lists of strings matching the shape of `difference_matrix`.
    """
    difference_matrix = np.array(difference_matrix)
    n_rows, n_cols = difference_matrix.shape
    result_strings = []
    for row_idx in range(n_rows):
        row_strings = []
        for col_idx in range(n_cols):
            value = difference_matrix[row_idx, col_idx]
            # Handle NaN or None values
            if np.isnan(value) or value is None:
                row_strings.append("NA")
                continue

            # Convert from percentage to [0, 1] range for format_value
            value_normalized = value / 100.0
            formatted_value = format_value(abs(value_normalized), significant_digits=significant_digits)

            if value >= 0:
                row_strings.append(f"+({formatted_value})")
            else:
                row_strings.append(f"-({formatted_value})")

        result_strings.append(row_strings)

    return result_strings


def print_results(args):
 

    # Folder containing the JSON files
    results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testing")

    # Define methods and evaluation types
    METHODS = ['zeroshot', 'coop', 'cocoop']
    EVAL_TYPES = ['base', 'novel']

    # Collect all results
    all_results = {}
    for method in METHODS:
        all_results[method] = {}

        results_file = os.path.join(results_folder, method, 'results.json')

        if os.path.exists(results_file):
            all_results[method] = json.load(open(results_file))
        else:
            raise ValueError(f"Results file {results_file} does not exist. Get results for Method='{method}'.")
    
        if method != 'zeroshot' and args.use_zebra:
            results_file = os.path.join(results_folder, method, 'results-zebra.json')
            if os.path.exists(results_file):
                all_results[method+'+zebra'] = json.load(open(results_file))
            else:
                raise ValueError(f"Results file {results_file} does not exist. Get results for Method='{method}'.")

    for seed in SEEDS:
        # Initialize table: rows are datasets, columns are method-evaltype combinations
        table_data = []
        column_headers = []

        if args.use_zebra: 
            METHODS_EXTENDED = [METHODS[0]] + METHODS[1:] + [method+'+zebra' for method in METHODS[1:]]
        else:
            METHODS_EXTENDED = METHODS

        # Build column headers
        for method in METHODS_EXTENDED:
            for eval_type in EVAL_TYPES:
                column_headers.append(f"{method.upper()}-{eval_type.upper()}")
        
        # Build rows for each dataset
        for dataset in DATASETS:
            row = [dataset]
            
            for method in METHODS_EXTENDED:
                for eval_type in EVAL_TYPES:
                    # Try to get result from results.json
                    try:
                        result = all_results[method].get(str(seed), {}).get(dataset, {}).get(eval_type)
                        if result is not None:
                            accuracy = result.get('accuracy', None)
                            if accuracy is not None: accuracy = float(accuracy)
                    except (KeyError, TypeError):
                        accuracy = None
                    
                    if accuracy is not None:
                        formatted_value = format_value(accuracy)  
                    else:
                        formatted_value = "NA"

                    row.append(formatted_value)

            table_data.append(row)
        
        # Add separator row before average
        separator_row = ["-----"] + ["-----"] * (len(table_data[0]) - 1)
        table_data.append(separator_row)
        
        # Add average row
        avg_row = ["AVG"]
        for col_idx in range(1, len(table_data[0])):
            col_values = []
            for row in table_data[:-1]:  # Exclude separator row
                try:
                    col_values.append(float(row[col_idx]))
                except (ValueError, TypeError):
                    pass
            if col_values:
                avg_value = np.mean(col_values)
                avg_row.append(format_value(avg_value / 100))  # divide by 100 since col_values are already in percentage form
            else:
                avg_row.append("NA")
        table_data.append(avg_row)
        
        
        table_data_numeric = get_table_data_numeric_from_table(table_data)
        table_data_numeric_avg = np.mean(table_data_numeric, axis=0, keepdims=True)

        difference_matrix = get_difference_matrix(table_data_numeric)
        difference_matrix_avg = np.mean(difference_matrix, axis=0, keepdims=True)
        
        difference_matrix_strings = difference_matrix_to_strings(difference_matrix)
        difference_matrix_strings_avg = difference_matrix_to_strings(difference_matrix_avg)[0]
        # Also get plus/minus formatted differences for alternate presentation
        difference_matrix_plusminus = difference_matrix_to_plusminus_strings(difference_matrix)
        difference_matrix_plusminus_avg = difference_matrix_to_plusminus_strings(difference_matrix_avg)[0]

        # Keep original table_data intact by working on copies
        original_table = [r.copy() for r in table_data]

        # Build table with LaTeX up/down triangles (no mutation of original_table)
        table_data_with_differences = []
        for i in range(len(original_table)):
            row = original_table[i].copy()
            if i == 11:
                continue
            if i == 12:
                diff_row = difference_matrix_strings_avg
            else:
                diff_row = difference_matrix_strings[i]
            for j in range(len(row)):
                if j > 2:
                    row[j] = row[j] + " " + diff_row[j-3]
            table_data_with_differences.append(row)

        # Build a second copy that uses +(...) / -(...) notation (also from original_table)
        table_data_with_plusminus = []
        for i in range(len(original_table)):
            row = original_table[i].copy()
            if i == 11:
                continue
            if i == 12:
                diff_row_pm = difference_matrix_plusminus_avg
            else:
                diff_row_pm = difference_matrix_plusminus[i]
            for j in range(len(row)):
                if j > 2:
                    row[j] = row[j] + " " + diff_row_pm[j-3]
            table_data_with_plusminus.append(row)

        

        latex_table = ""
        for i, row in enumerate(table_data_with_differences):
            latex_table += " & ".join(row) + " \\\\\n" 
            if i==10: latex_table += "\\midrule\n"
        print(latex_table)


        
        # Print original table (unchanged markers)
        print(f"\n\n########## SEED {seed} - ACCURACY TABLE (ORIGINAL) ##########\n")
        print(tabulate(table_data, headers=["Dataset ↓ . Method →"] + column_headers, tablefmt="simple"))
        print("\n")

        # Print alternate table with +(...) / -(...) diffs
        print(f"\n\n########## SEED {seed} - ACCURACY TABLE (WITH +/-(DIFFS)) ##########\n")
        print(tabulate(table_data_with_plusminus, headers=["Dataset ↓ . Method →"] + column_headers, tablefmt="simple"))
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print Results")
    parser.add_argument('--method', type=str, default='palm', help='Method name (zeroshot, coop, cocoop)')
    parser.add_argument('--use_zebra', action='store_true', help='Use ZEBRA')
    args = parser.parse_args()


    # Datasets and number of folds
    DATASETS = [
                'Beijing-Opera',
                'CREMA-D',
                'ESC50-Actions',
                'ESC50',
                'GT-Music-Genre',
                'NS-Instruments',
                'RAVDESS',
                'SESA',
                'TUT2017',
                'UrbanSound8K',
                'VocalSound',
            ]

    SEEDS = [0]
    # SEEDS = [0,1,2]

    print_results(args)