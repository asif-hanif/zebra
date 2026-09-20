import os
import torch
import numpy as np
from tqdm import tqdm
import json

from .utils import get_scores, print_scores, save_scores, timeit, save_model, get_save_model_path, compute_loss, orthogonal_loss


def run_epoch(model, dataloader, optimizer, device, args=None):
    model.train()
    # model.audio_encoder.eval()
    # model.text_encoder.eval()

    losses = []
    actual_labels = []
    predicted_labels = []

    for i, (audio, label) in enumerate(dataloader):

        audio = audio.to(device).squeeze(1)
        label = label.to(device)
 

        logits = model(audio)
        loss = compute_loss(logits, label, args=args)
        # loss = loss + 10.0*orthogonal_loss(model.prompt_learner.ctx)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

        actual_labels.extend(label.cpu().numpy())
        predicted_labels.extend(logits.argmax(axis=1).cpu().numpy())

    avg_loss = sum(losses) / len(losses)

    return avg_loss, actual_labels, predicted_labels


@timeit
def run_evaluation(model, dataloader, device, base_or_novel_or_all, args=None):
    model.eval()

    losses = []
    actual_labels = []
    predicted_labels = []

    print(f"\n\nEvaluating the model ({base_or_novel_or_all.upper()} only) ...")
    
    with torch.no_grad():
        for i, (audio, label) in enumerate(dataloader):
        # for i, (audio, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            print(f"Batch {i+1}/{len(dataloader)}")

            audio = audio.to(device).squeeze(1)
            label = label.to(device)
            
            logits = model(audio)
            loss = compute_loss(logits, label, args=args)

            losses.append(loss.item())

            actual_labels.extend(label.cpu().numpy())
            predicted_labels.extend(logits.argmax(axis=1).cpu().numpy())
            
            # break
    avg_loss = sum(losses) / len(losses)

    return avg_loss, actual_labels, predicted_labels


@timeit
def run_training(model, train_dataloader, test_dataloader, optimizer, device, epochs=50, args=None):

    for epoch in tqdm(range(epochs), total=epochs):

        train_loss, actual_labels, predicted_labels = run_epoch(model, train_dataloader, optimizer, device, args=args)

        if (epoch+1)%5 == 0:
            accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classnames)
            print(f"\n\n-------------------------------\nTrain Evaluation (Epoch {epoch + 1}/{epochs} -- {args.eval_type.upper()})\n-------------------------------\n")
            print_scores(accuracy, f1_score, precision, recall, train_loss) 
            

        if (epoch+1)%args.freq_test_model == 0:
            if args.eval_last_epoch_only and (epoch != epochs-1):
                continue

            test_loss, actual_labels, predicted_labels = run_evaluation(model, test_dataloader, device, args.eval_type, args=args)
            accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classnames)
            print(f"\n\n-------------------------------\nTest Evaluation -- {args.eval_type.upper()}\n-------------------------------\n")
            print_scores(accuracy, f1_score, precision, recall, test_loss)

            if (epoch == epochs-1) and args.do_logging:
                print("\n\nFinal Evaluation")
                print("Saving Results ...")
                save_scores(args.seed, epoch, accuracy, f1_score, precision, recall, test_loss, args.json_file_path)
                print("Results Saved\n\n")
        

    if args.save_model:
        save_model_path = get_save_model_path(args)
        save_model(args, model, save_model_path)
        print(f"Model saved to {save_model_path}")
        # # Save L1 norm of all learnable and non-learnable parameters and buffers
        # save_l1_norm_model_parameters(model, save_model_path)


