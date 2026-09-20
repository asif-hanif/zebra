import os
import json
import argparse
import pandas as pd
import random
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)



def split_base_novel_classnames(dataset, args):

    root = os.path.join(args.dataset_root, dataset)

    if dataset in ['Beijing-Opera', 'ESC50-Actions', 'ESC50', 'TUT2017', 'UrbanSound8K']:
        df_train = pd.read_csv(os.path.join(root, "csv_files", f"train.csv"))
        df_test = pd.read_csv(os.path.join(root, "csv_files", f"test.csv"))
    else:
        df_train = pd.read_csv(os.path.join(root, f"train.csv"))
        df_test = pd.read_csv(os.path.join(root, f"test.csv"))

    assert set(df_train['classname'].unique().tolist()) == set(df_test['classname'].unique().tolist()), "Classnames in train and test datasets are different."


    classnames = df_train['classname'].unique().tolist()
    classnames.sort()

    shuffled_classnames = random.sample(classnames, len(classnames))
    num_classes = len(shuffled_classnames)
    split_point = num_classes // 2
    base_classnames = shuffled_classnames[:split_point]
    novel_classnames = shuffled_classnames[split_point:]

    base_classnames.sort()
    novel_classnames.sort()

    return {'base': base_classnames, 'novel': novel_classnames, 'all': classnames}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Base and Novel Class Splitter")
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to the dataset root directory')
    args = parser.parse_args()

    seed_everything(42)

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

    for dataset in DATASETS:
        print(f"\nBase-Novel Split for Dataset: {dataset} ...")
        base_novel_classnames = split_base_novel_classnames(dataset, args)
        save_path = os.path.join('scripts', 'base-novel-classes', f'{dataset}_BaseNovelClassnames.json')
        
        with open(save_path, 'w') as f:
            json.dump(base_novel_classnames, f, indent=2)
        print(f"Base and Novel classnames saved to {save_path}.\n")

    