import os
import sys
import json
import pytz
import argparse
import datetime
import pandas as pd
import random
from sklearn.metrics import classification_report

import torch
import torch.nn.functional as F

from .dataset import FewShotDataset

METHODS = ['zeroshot', 'coop', 'cocoop', 'palm']

def get_model(args, pengi, methods):
    print(f"Using Method: '{args.method_name.upper()}'\n")

    if args.method_name == 'zeroshot':
        model = methods.ZeroShot(args, pengi)
    elif args.method_name == 'coop':
        model = methods.COOP(args, pengi)
    elif args.method_name == 'cocoop':
        model = methods.COCOOP(args, pengi)
    elif args.method_name == 'palm':
        model = methods.PALM(args, pengi)
        # raise NotImplementedError("Model 'palm' is not implemented yet.")
    else:
        raise ValueError(f"Model '{args.method_name}' is not supported. Choose from: [{', '.join(METHODS)}]")
    
    return model

def entropy_loss(logits, eps=1e-8):
    """
    logits: Tensor of shape [B, C]
    returns: mean entropy over batch (scalar)
    """
    p = F.softmax(logits, dim=-1)
    entropy = -(p * torch.log(p + eps)).sum(dim=-1)
    return entropy.mean()

cross_entropy_loss = torch.nn.CrossEntropyLoss()


def compute_loss(logits, labels, args):
    ce_loss = cross_entropy_loss(logits, labels)
    if args.use_zebra:
        ent_loss = entropy_loss(logits)
        total_loss = ce_loss - 0.05 * ent_loss  # maximize entropy
        return total_loss
    else:
        return ce_loss


def orthogonal_loss(ctx):
    """
    ctx: Tensor of shape [n_ctx, ctx_dim]
    returns: orthogonal loss (scalar)
    """
    ctx = ctx @ ctx.t() + 0.0
    ctx = ctx / ctx.norm(dim=1, keepdim=True) 
    
    I = torch.eye(ctx.size(0), device=ctx.device)

    loss_ortho = ((ctx - I) ** 2).mean()

    return loss_ortho

def get_dataloaders(args):
    print("\n\nLoading Base and Novel Classnames ...\n")
    
    base_novel_all_classnames = json.load(open(os.path.join('scripts', 'base-novel-classes', f'{args.eval_dataset}_BaseNovelClassnames.json')))
    args.base_novel_all_classnames = base_novel_all_classnames

    train_dataset = FewShotDataset(args.dataset_root, 'train' , num_shots=args.num_shots, repeat=args.repeat , process_audio_fn=args.process_audio_fn, resample=args.resample, classnames=base_novel_all_classnames['base'])
    
    if args.eval_type == 'base': 
        test_dataset  = FewShotDataset(args.dataset_root, 'test'  , num_shots=-1, repeat=args.repeat , process_audio_fn=args.process_audio_fn, resample=args.resample, classnames=base_novel_all_classnames['base'])
    elif args.eval_type == 'novel': 
        test_dataset  = FewShotDataset(args.dataset_root, 'test'  , num_shots=-1, repeat=args.repeat , process_audio_fn=args.process_audio_fn, resample=args.resample, classnames=base_novel_all_classnames['novel'])
    elif args.eval_type == 'all':
        test_dataset  = FewShotDataset(args.dataset_root, 'test'  , num_shots=-1, repeat=args.repeat , process_audio_fn=args.process_audio_fn, resample=args.resample, classnames=base_novel_all_classnames['all'])
    else:
        raise ValueError(f"Evaluation type '{args.eval_type}' is not supported. Choose from: ['base', 'novel', 'all']")
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader


def save_model(args, model, save_model_path):
    print(f"Saving Context Weights for Method: '{args.method_name.upper()}'\n")
    if args.method_name in ['coop', 'cocoop', 'palm']:
        checkpoint = {'prompt_learner': model.prompt_learner.state_dict()}
        checkpoint['pengi_bn0_buffer'] = {'running_mean': model.audio_encoder.base.htsat.bn0.running_mean.clone(),
                                            'running_var': model.audio_encoder.base.htsat.bn0.running_var.clone(),
                                            'num_batches_tracked': model.audio_encoder.base.htsat.bn0.num_batches_tracked.clone()}
        checkpoint['base_novel_all_classnames'] = args.base_novel_all_classnames
        torch.save(checkpoint, save_model_path)
    else:
        raise ValueError(f"Model '{args.method_name}' is not supported. Choose from: [{', '.join(METHODS)}]")


    
def load_model(args, model):
        load_model_path = get_load_model_path(args)
        print(f"Loading Model (Prompt Learner) from: {load_model_path}\n\n")
        checkpoint = torch.load(load_model_path)

        if args.method_name == 'coop':
            model.prompt_learner.ctx.data.copy_(checkpoint['prompt_learner']['ctx'])
            # model.prompt_learner.sos.copy_(checkpoint['prompt_learner']['sos'])
            # model.prompt_learner.classes_pad.copy_(checkpoint['prompt_learner']['classes_pad'])
        elif args.method_name == 'cocoop':
            model.prompt_learner.ctx.data.copy_(checkpoint['prompt_learner']['ctx'])
            # model.prompt_learner.sos.copy_(checkpoint['prompt_learner']['sos'])
            # model.prompt_learner.classes_pad.copy_(checkpoint['prompt_learner']['classes_pad'])
            # Extract meta_net weights from checkpoint (keys have 'meta_net.' prefix)
            meta_net_state = {k.replace('meta_net.', ''): v for k, v in checkpoint['prompt_learner'].items() if k.startswith('meta_net.')}
            model.prompt_learner.meta_net.load_state_dict(meta_net_state)
        # elif args.method_name == 'palm':
        #     # ctx is a Linear layer, use load_state_dict for proper loading
        #     prompt_learner_state = checkpoint['prompt_learner'].copy()
        #     model.prompt_learner.load_state_dict(prompt_learner_state, strict=False)
        # elif args.method_name == 'palm':
        #     model.prompt_learner.ctx.data.copy_(checkpoint['prompt_learner']['ctx'])
        #     model.prompt_learner.ctx_audio.weight.data.copy_(checkpoint['prompt_learner']['ctx_audio.weight'])
        elif args.method_name == 'palm':
            model.prompt_learner.ctx.data.copy_(checkpoint['prompt_learner']['ctx'])
            meta_net_state = {k.replace('meta_net.', ''): v for k, v in checkpoint['prompt_learner'].items() if k.startswith('meta_net.')}
            model.prompt_learner.meta_net.load_state_dict(meta_net_state)
        else:
            raise ValueError(f"Model '{args.method_name}' is not supported. Choose from: [{', '.join(METHODS)}]")
        
        # if args.eval_type == 'base':
        model.audio_encoder.base.htsat.bn0.running_mean.copy_(checkpoint['pengi_bn0_buffer']['running_mean'])
        model.audio_encoder.base.htsat.bn0.running_var.copy_(checkpoint['pengi_bn0_buffer']['running_var'])
        model.audio_encoder.base.htsat.bn0.num_batches_tracked.copy_(checkpoint['pengi_bn0_buffer']['num_batches_tracked'])
        # raise NotImplementedError("\n\nLoading model is not implemented yet.\n\n")



def get_save_model_path(args):
    if args.use_zebra:
        use_zebra = "-ZEBRA"
    else:
        use_zebra = ""

    if args.save_model_abs_path is None: # create a default path and save model to it
        save_model_path = os.path.join(args.save_model_path, args.method_name)
        if not os.path.exists(save_model_path): os.mkdir(save_model_path)
        save_model_path = os.path.join(save_model_path, f"{args.exp_name+'-SEED'+str(args.seed)}{use_zebra}.pth")
        return save_model_path
    else:   # save model to the specified path
        return args.save_model_abs_path

def get_load_model_path(args):
    if args.use_zebra:
        use_zebra = "-ZEBRA"
    else:
        use_zebra = ""

    if args.load_model_abs_path is not None:
        load_model_path = args.load_model_abs_path
    else:
        load_model_path = os.path.join(args.load_model_path, args.method_name, f"{args.exp_name+'-SEED'+str(args.seed)}{use_zebra}.pth")
    
    if args.method_name != "zeroshot" and not os.path.exists(load_model_path): 
        raise ValueError(f"Model file '{load_model_path}' does not exist. Specify the correct path to the model file.")
    
    return load_model_path


def get_args():
    parser = argparse.ArgumentParser(description='PALM: Prompt-based Few-Shot Learning for Audio Language Models')
    parser.add_argument('--method_name', type=str, default='', help='Model Name (default: None)', required=True)
    parser.add_argument('--use_zebra', help='Use ZEBRA (default: False)', action='store_true')
    parser.add_argument('--save_model', help='Save the trained model (default: False)', action='store_true')
    parser.add_argument('--save_model_path', type=str, default=None, help='Path to save the trained model (default: None)')
    parser.add_argument('--save_model_abs_path', type=str, default=None, help='Absolute path to save the trained model (default: None)')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to the pre-trained model (learnable context) weights (default: None)')
    parser.add_argument('--load_model_abs_path', type=str, default=None, help='Absolute path to the pre-trained model (learnable context) weights (default: None)')
    parser.add_argument('--source_model_dataset', type=str, default=None, help='Dataset used to train the source model (default: None)')
    parser.add_argument('--eval_dataset', type=str, default=None, help='Dataset to evaluate the model on (default: None)')
    parser.add_argument('--dataset_root', type=str, default='', help='Path to the dataset root directory (default: None)', required=True)
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch (default: 0)')
    parser.add_argument('--freq_test_model', type=int, default=10, help='Frequency of testing the model (default: 10)')
    parser.add_argument('--eval_last_epoch_only', default=True, help='Evaluate only at the last epoch (default: True)')
    parser.add_argument('--spec_aug', help='Apply Spectrogram Augmentation (default: False)', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed (default: 0)')
    parser.add_argument('--eval_only', help='Evaluate the model only (default: False)', action='store_true')
    parser.add_argument('--eval_type', type=str, default='base', help="Evaluate on 'base', 'novel' or 'all' classes (default: 'novel')")
    parser.add_argument('--exp_name', type=str, default='', help='experiment name', required=True)
    parser.add_argument('--do_logging', help='Disable Logging (default: False)', action='store_true')
    parser.add_argument('--prompt_prefix', type=str, default='The is a recording of ', help='Prompt Prefix (default: The is a recording of )')

    # COOP/COCOOP and PALM Arguments
    parser.add_argument('--n_ctx', type=int, default=16, help='Number of context tokens (default: 16)')
    parser.add_argument('--ctx_dim', type=int, default=512, help='Dimension of the context vector (default: 512)')

    # Few-Shot Learning Arguments
    parser.add_argument('--num_shots', type=int, default=16, help='Number of shots (default: 16)')
    parser.add_argument('--resample', type=bool, default=True, help='Resample samples if needed (default: True)')
    parser.add_argument('--repeat', type=bool, default=False, help='Repeat samples if needed (default: False)')
    parser.add_argument('--base_novel_all_classnames', type=dict, default=None, help='Base, Novel, All classnames for test split (default: None)')
    
    args = parser.parse_args()

    # Sanity check on Arguments
    if not os.path.exists(args.dataset_root):
        raise ValueError(f"\n\nDirectory '{args.dataset_root}' does not exist. Specify the correct path to the dataset.\n\n")
    if args.save_model and args.save_model_abs_path is None:
        save_model_path = args.save_model_path
        if not os.path.exists(save_model_path): raise ValueError(f"\n\nDirectory '{save_model_path}' does not exist. Create or specify the correct the directory to save the trained model.\n\n")
    elif args.save_model and args.save_model_abs_path is not None:
        base_directory = os.path.dirname(args.save_model_abs_path)
        if not os.path.exists(base_directory): raise ValueError(f"\n\nDirectory '{base_directory}' does not exist. Create or specify the correct the directory to save the trained model.\n\n")
    else:
        pass

    if args.method_name == 'zeroshot': args.eval_only = True

    if args.eval_only and args.method_name != "zeroshot":
        load_model_path = get_load_model_path(args)
        if args.method_name != "zeroshot" and not os.path.exists(load_model_path): raise ValueError(f"\n\nEvaluation Mode: Model file '{load_model_path}' does not exist. Specify the correct path to the model file.\n\n")
      
    return args



def print_total_time(now_start, now_end):
	print(f'\nEnd Time & Date = {now_end.strftime("%I:%M %p")} , {now_end.strftime("%d_%b_%Y")}\n')
	duration_in_s = (now_end - now_start).total_seconds()
	days  = divmod(duration_in_s, 86400)   # Get days (without [0]!)
	hours = divmod(days[1], 3600)          # Use remainder of days to calc hours
	minutes = divmod(hours[1], 60)         # Use remainder of hours to calc minutes
	seconds = divmod(minutes[1], 1)        # Use remainder of minutes to calc seconds
	print(f"Total Time => {int(days[0])} Days : {int(hours[0])} Hours : {int(minutes[0])} Minutes : {int(seconds[0])} Seconds\n\n")




def print_dataset_info(train_dataloader, test_dataloader):
	n_classes = train_dataloader.dataset.n_classes
	num_batches_train = len(train_dataloader)
	num_batches_test = len(test_dataloader)

	print("\n########################\nDataset Information\n########################\n")
	print("Length of the Train Dataset: ", len(train_dataloader.dataset))
	print("Length of the Test Dataset: ", len(test_dataloader.dataset))
	print("Train Batch Size: ", train_dataloader.batch_size)
	print("Test Batch Size: ", test_dataloader.batch_size)
	print("Number of Batches in Train Dataloader: ", num_batches_train)
	print("Number of Batches in Test Dataloader: ", num_batches_test)
	print("Number of Classes: ", n_classes)
     

def get_scores(actual_labels, predicted_labels, classnames):
    cls_report = classification_report(actual_labels, predicted_labels, target_names=classnames, output_dict=True)
    accuracy = cls_report['accuracy']
    f1_score = cls_report['macro avg']['f1-score']
    precision = cls_report['macro avg']['precision']
    recall = cls_report['macro avg']['recall']
    return accuracy, f1_score, precision, recall


def print_scores(accuracy, f1_score, precion, recall, avg_loss):
    print(f"{'Accuracy':<15} = {accuracy:0.4f}")
    print(f"{'F1-Score':<15} = {f1_score:0.4f}")
    print(f"{'Precision':<15} = {precion:0.4f}")
    print(f"{'Recall':<15} = {recall:0.4f}")
    print(f"{'Average Loss':<15} = {avg_loss:0.4f}\n\n")


def save_scores(seed, epoch, accuracy, f1_score, precision, recall, avg_loss, json_file_path):
    if not os.path.exists(json_file_path):
        # create the file if it doesn't exist
        with open(json_file_path, "w") as file:
            file.write("{}")
        
    # load existing results
    with open(json_file_path, "r") as file:
        scores_json = json.load(file)

    scores_json[f"seed_{seed}"] = {"accuracy": f"{accuracy:0.4f}", "f1_score": f"{f1_score:0.4f}", "precision": f"{precision:0.4f}", "recall": f"{recall:0.4f}", "avg_loss": f"{avg_loss:0.4f}", "epoch": epoch}

    for metric in scores_json[f"seed_{seed}"].keys():
        if metric != 'epoch': scores_json[f"seed_{seed}"][metric] = float(scores_json[f"seed_{seed}"][metric])


    # save updated results
    with open(json_file_path, "w") as file:
        json.dump(scores_json, file, indent=2) 



# Decorator to measure the time taken by a function
def timeit(func):
    import time
    def wrapper(*args, **kwargs):
        
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        
        duration_in_s = end - start
        days  = divmod(duration_in_s, 86400)   # Get days (without [0]!)
        hours = divmod(days[1], 3600)          # Use remainder of days to calc hours
        minutes = divmod(hours[1], 60)         # Use remainder of hours to calc minutes
        seconds = divmod(minutes[1], 1)        # Use remainder of minutes to calc seconds


        date_now = datetime.datetime.now(pytz.timezone('Asia/Dubai'))
        print(f'\n\nTime & Date = {date_now.strftime("%I:%M %p")} , {date_now.strftime("%d_%b_%Y")}  GST')
        print(f"\nTotal Time => {int(days[0])} Hours : {int(minutes[0])} Minutes : {int(seconds[0])} Seconds\n\n")
        return result
        
    return wrapper


############################################################################################################
# Logging Functions
############################################################################################################


# Define a Tee class to duplicate output to both stdout and a log file
class Tee:
    def __init__(self, *files):
        self.files = files
 
    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()
 
    def flush(self):
        for file in self.files:
            file.flush()

# Define a function to redirect stdout and stderr to a log file
def redirect_output_to_log(log_file):
    # Open the log file in append mode
    log = open(log_file, 'a')
 
    # Duplicate stdout and stderr
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)

    return log

# Define a function to setup logging
def setup_logging(args):
    log_dir = os.path.join('logs', "testing" if args.eval_only else "training", args.method_name) # log file dir
    args.log_dir = log_dir

    if args.use_zebra:
        use_zebra = "-ZEBRA"
    else:
        use_zebra = ""
    
    if args.do_logging:
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        if args.eval_only:
            log_file_path = os.path.join(log_dir, f"{args.exp_name}-SEED{args.seed}-{args.eval_type.upper()}{use_zebra}.log")            
            json_file_path = log_file_path.replace('.log', '.json')
        else:
            log_file_path = os.path.join(log_dir, f"{args.exp_name}-SEED{args.seed}-{args.eval_type.upper()}{use_zebra}.log")            
            json_file_path = log_file_path.replace('.log', '.json')

        if os.path.exists(log_file_path): os.remove(log_file_path)
        if os.path.exists(json_file_path): os.remove(json_file_path)
        
        args.json_file_path = json_file_path
        print(f"\nLogging to '{log_file_path}'\n")
        log_file = redirect_output_to_log(log_file_path) # redirect terminal output to log file
    else:
        log_file =None

    return log_file

