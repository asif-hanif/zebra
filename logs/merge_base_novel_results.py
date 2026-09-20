import os
import json
import numpy as np
import argparse


# Function to load JSON data from a file
def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


# def check_seed_existence(results):
#     seeds_exist = []
#     for seed in SEEDS:
#         if f'seed_{seed}' in results.keys(): seeds_exist.append(seed)
#     return seeds_exist


# Function to get results for all seeds of a dataset and method   
# def get_dataset_results(method, dataset, seed, results_folder, eval_type):

#     json_path = os.path.join(results_folder, f"{dataset}-SEED{seed}-{eval_type.upper()}.json")
#     if os.path.exists(json_path):
#         results = load_json(json_path)[f"seed_{seed}"]
#     else:
#         raise ValueError(f"File {json_path} does not exist. Get results for Method='{method}', Dataset='{dataset}', Seed='{seed}', EvalType='{eval_type}'.") 
    
#     return results



def get_results(method, results_folder, use_zebra):
    results = {}
    for seed in SEEDS:
        results[seed] = {}
        for dataset in DATASETS:
            json_path_base = os.path.join(results_folder, f"{dataset}-SEED{seed}-BASE{'-ZEBRA' if use_zebra else ''}.json")
            json_path_novel = os.path.join(results_folder, f"{dataset}-SEED{seed}-NOVEL{'-ZEBRA' if use_zebra else ''}.json")

            results[seed][dataset] = {
                "base": load_json(json_path_base)[f"seed_{seed}"],
                "novel": load_json(json_path_novel)[f"seed_{seed}"]
            }
    return results



def merge_results(args):
        
    method = args.method
    use_zebra = args.use_zebra

    # Folder containing the JSON files
    results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testing", method)
    
    results = get_results(method, results_folder, use_zebra) # Structure: results[seed][dataset]={base: ..., novel: ..., }
    
    result_path = os.path.join(results_folder, f"results{'-zebra' if use_zebra else ''}.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved in {result_path} file.\n\n")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Print Results")
    parser.add_argument('--method', type=str, default='palm', help='Method name (zeroshot, coop, cocoop)')
    parser.add_argument('--use_zebra', action='store_true', help='Use ZEBRA')
    args = parser.parse_args()

    print(f"\n\nProcessing results for method: {args.method} {'with ZEBRA' if args.use_zebra else 'without ZEBRA'} ...\n\n")


    # Datasets 
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



    # methods = ['zeroshot', 'coop', 'cocoop', 'palm']
    # methods = ['coop', 'cocoop', 'palm']
    # methods = ['palm']


    SEEDS = [0]
    # SEEDS = [0,1,2]

    merge_results(args)