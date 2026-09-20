import os
import random
import numpy as np
import datetime
import pytz
from tqdm import tqdm
from pprint import pprint

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore', message='.*torch.meshgrid.*')
warnings.filterwarnings('ignore', message='.*resume_download.*')


import methods
from pengi import pengi


from utils import trainer
from utils.utils import print_total_time, get_args, get_dataloaders, get_model, setup_logging, get_scores, print_scores, save_scores, load_model

# to solve  the issue of : the current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def main(args):

    # warnings_filter()

    print(f"\n\n{'METHOD:':<10}{args.method_name.upper()}")
    if args.use_zebra:
        print(f"{'USE ZEBRA:':<10}True")
    else:
        print(f"{'USE ZEBRA:':<10}False")
    print(f"{'Dataset:':<10}{args.dataset_root.split('/')[-1]}")
    print(f"{'Seed:':<10}{args.seed}\n\n")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    args.process_audio_fn = pengi.preprocess_audio

    # to ensure reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
    train_dataloader, test_dataloader = get_dataloaders(args)
    if args.eval_only: args.classnames = test_dataloader.dataset.classnames
    else: args.classnames = train_dataloader.dataset.classnames
 
    model = get_model(args, pengi, methods)
    model.to(device)

    # criterion = nn.CrossEntropyLoss()
    
    print("\nArguments:\n")
    for arg in vars(args): print(f"{arg:<25}: {getattr(args, arg)}")
    print("\n\n")


    if args.eval_only:
        if args.method_name != "zeroshot": load_model(args, model)
        test_loss, actual_labels, predicted_labels = trainer.run_evaluation(model, test_dataloader, device, args.eval_type, args=args)
        accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classnames)
        print(f"\n\n-------------------------------\nTest Evaluation -- {args.eval_type.upper()}\n-------------------------------\n")
        print_scores(accuracy, f1_score, precision, recall, test_loss)
        if args.do_logging:
            print("Saving Results ...") 
            save_scores(args.seed, -1, accuracy, f1_score, precision, recall, test_loss, args.json_file_path)
            print("Results Saved\n\n")
    else:
        #optimizer = torch.optim.Adam(model.prompt_learner.parameters(), lr=args.lr)
        optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=args.lr, momentum=0.9)
        trainer.run_training(model, train_dataloader, test_dataloader, optimizer, device, epochs=args.n_epochs, args=args)



if __name__ == "__main__":

    args = get_args()
    log_file = setup_logging(args)

    print("\n\n##############################################")
    print("PALM: Prompt Learning in Audio Language Models")
    print("##############################################\n\n")
    date_now = datetime.datetime.now(pytz.timezone('Asia/Dubai'))
    print(f'Time & Date = {date_now.strftime("%I:%M %p")} , {date_now.strftime("%d_%b_%Y")}  GST\n')

    main(args)

