
import csv
import os
import argparse
import random
import logging
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

from utils import __balance_val_split, __split_of_train_sequence, __log_class_statistics
from datasets.czech_slr_dataset import CzechSLRDataset
from spoter.spoter_model import SPOTER
from spoter.utils import train_epoch, evaluate
from spoter.gaussian_noise import GaussianNoise


def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="lsa_64_spoter",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--num_classes", type=int, default=64, help="Number of classes to be recognized by the model")
    parser.add_argument("--hidden_dim", type=int, default=108,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")

    # Data
    parser.add_argument("--training_set_path", type=str, default="", help="Path to the training dataset CSV file")
    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")
    parser.add_argument("--experimental_train_split", type=float, default=None,
                        help="Determines how big a portion of the training set should be employed (intended for the "
                             "gradually enlarging training set experiment from the paper)")

    parser.add_argument("--validation_set", type=str, choices=["from-file", "split-from-train", "none"],
                        default="from-file", help="Type of validation set construction. See README for further rederence")
    parser.add_argument("--validation_set_size", type=float,
                        help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
                             " to 'split-from-train'")
    parser.add_argument("--validation_set_path", type=str, default="", help="Path to the validation dataset CSV file")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the model training")
    parser.add_argument("--log_freq", type=int, default=1,
                        help="Log frequency (frequency of printing all the training info)")

    # Checkpointing
    parser.add_argument("--save_checkpoints", type=bool, default=True,
                        help="Determines whether to save weights checkpoints")

    # Scheduler
    parser.add_argument("--scheduler_factor", type=int, default=0.1, help="Factor for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for the ReduceLROnPlateau scheduler")

    # Gaussian noise normalization
    parser.add_argument("--gaussian_mean", type=int, default=0, help="Mean parameter for Gaussian noise layer")
    parser.add_argument("--gaussian_std", type=int, default=0.001,
                        help="Standard deviation parameter for Gaussian noise layer")

    # Visualization
    parser.add_argument("--plot_stats", type=bool, default=True,
                        help="Determines whether continuous statistics should be plotted at the end")
    parser.add_argument("--plot_lr", type=bool, default=True,
                        help="Determines whether the LR should be plotted at the end")
    # Adding momentum and weight decay
    parser.add_argument("--momentum", type=float, default=0, help="Momentum for the loss plot")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for the loss plot")
    return parser

def train_findlr(args):
    
    # MARK: TRAINING PREPARATION AND MODULES

    # Initialize all the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Set the output format to print into the console and save into LOG file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + ".log")
        ]
    )

    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Construct the model
    slrt_model = SPOTER(num_classes=args.num_classes, hidden_dim=args.hidden_dim)
    slrt_model.train(True)
    slrt_model.to(device)

    #  Print the model summary
    # Print number of trainable parameters

    logging.info(slrt_model)
    # print(slrt_model)

    total_params = sum(p.numel() for p in slrt_model.parameters() if p.requires_grad)
    logging.info("Total number of trainable parameters: {}".format(total_params))
    print("Total number of trainable parameters: {}".format(total_params))
    # Construct the other modules
    cel_criterion = nn.CrossEntropyLoss()
    
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, factor=args.scheduler_factor, patience=args.scheduler_patience)

    # Ensure that the path for checkpointing and for images both exist
    Path("out-checkpoints/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    Path("out-img/").mkdir(parents=True, exist_ok=True)


    # MARK: DATA

    # Training set
    transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    train_set = CzechSLRDataset(args.training_set_path, transform=transform, augmentations=True)

    # Validation set
    if args.validation_set == "from-file":
        val_set = CzechSLRDataset(args.validation_set_path)
        val_loader = DataLoader(val_set, shuffle=True, generator=g)

    elif args.validation_set == "split-from-train":
        train_set, val_set = __balance_val_split(train_set, 0.2)

        val_set.transform = None
        val_set.augmentations = False
        val_loader = DataLoader(val_set, shuffle=True, generator=g)

    else:
        val_loader = None

    # Testing set
    if args.testing_set_path:
        eval_set = CzechSLRDataset(args.testing_set_path)
        eval_loader = DataLoader(eval_set, shuffle=True, generator=g)

    else:
        eval_loader = None

    # Final training set refinements
    if args.experimental_train_split:
        train_set = __split_of_train_sequence(train_set, args.experimental_train_split)

    train_loader = DataLoader(train_set, shuffle=True, generator=g)


    # MARK: TRAINING
    train_acc, val_acc = 0, 0
    losses, train_accs, val_accs = [], [], []
    lr_progress = []
    top_train_acc, top_val_acc = 0, 0
    checkpoint_index = 0

    if args.experimental_train_split:
        print("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
        logging.info("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")

    else:
        print("Starting " + args.experiment_name + "...\n\n")
        logging.info("Starting " + args.experiment_name + "...\n\n")
    cur_lr = 1e-10
    for epoch in range(args.epochs):
        #  update the learning rate
        cur_lr = cur_lr * 4
        sgd_optimizer = optim.SGD(slrt_model.parameters(), lr=cur_lr, momentum = args.momentum, weight_decay=args.weight_decay)
        train_loss, _, _, train_acc = train_epoch(slrt_model, train_loader, cel_criterion, sgd_optimizer, device)
        losses.append(train_loss.item() / len(train_loader))
        lr_progress.append(cur_lr)
        train_accs.append(train_acc)

        if val_loader:
            slrt_model.train(False)
            _, _, val_acc = evaluate(slrt_model, val_loader, device)
            slrt_model.train(True)
            val_accs.append(val_acc)

        # # Save checkpoints if they are best in the current subset
        # if args.save_checkpoints:
        #     if train_acc > top_train_acc:
        #         top_train_acc = train_acc
        #         torch.save(slrt_model, "out-checkpoints/" + args.experiment_name + "/checkpoint_t_" + str(checkpoint_index) + ".pth")

        #     if val_acc > top_val_acc:
        #         top_val_acc = val_acc
        #         torch.save(slrt_model, "out-checkpoints/" + args.experiment_name + "/checkpoint_v_" + str(checkpoint_index) + ".pth")

        if epoch % args.log_freq == 0:
            print("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item() / len(train_loader)) + " acc: " + str(train_acc))
            logging.info("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item() / len(train_loader)) + " acc: " + str(train_acc))

            if val_loader:
                print("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))
                logging.info("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))

            print("")
            logging.info("")

        # Reset the top accuracies on static subsets
        if epoch % 10 == 0:
            top_train_acc, top_val_acc = 0, 0
            checkpoint_index += 1

    # MARK: TESTING

    print("\nTesting checkpointed models starting...\n")
    logging.info("\nTesting checkpointed models starting...\n")

    top_result, top_result_name = 0, ""

    if eval_loader:
        for i in range(checkpoint_index):
            for checkpoint_id in ["t", "v"]:
                # tested_model = VisionTransformer(dim=2, mlp_dim=108, num_classes=100, depth=12, heads=8)
                tested_model = torch.load("out-checkpoints/" + args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i) + ".pth")
                tested_model.train(False)
                _, _, eval_acc = evaluate(tested_model, eval_loader, device, print_stats=True)

                if eval_acc > top_result:
                    top_result = eval_acc
                    top_result_name = args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i)

                print("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))
                logging.info("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))

        print("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
        logging.info("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")


    # PLOT 0: Performance (loss, accuracies) chart plotting
    if args.plot_stats:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(losses) + 1), losses, c="#D64436", label="Training loss")
        ax.plot(range(1, len(train_accs) + 1), train_accs, c="#00B09B", label="Training accuracy")

        if val_loader:
            ax.plot(range(1, len(val_accs) + 1), val_accs, c="#E0A938", label="Validation accuracy")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
        ax.grid()

        fig.savefig("out-img/" + args.experiment_name + "_loss.png")

    # PLOT 1: Learning rate progress
    if args.plot_lr:
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
        ax1.set(xlabel="Epoch", ylabel="LR", title="")
        ax1.grid()

        fig1.savefig("out-img/" + args.experiment_name + "_lr.png")

    print("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    logging.info("\nAny desired statistics have been plotted.\nThe experiment is finished.")

    # Plot learning rate vs loss, learning rate vs accuracy
    fig2, ax2 = plt.subplots()
    ax2.plot(lr_progress, losses, label="Loss")
    ax2.set(xlabel="LR", ylabel="Loss", title="")
    ax2.grid()

    fig2.savefig("out-img/" + args.experiment_name + "_findlr_lr_loss.png")

    # Plot learning rate vs accuracy
    fig3, ax3 = plt.subplots()
    ax3.plot(lr_progress, train_accs, label="Training accuracy")
    ax3.set(xlabel="LR", ylabel="Accuracy", title="")
    ax3.grid()

    fig3.savefig("out-img/" + args.experiment_name + "_findlr_lr_acc.png")



def train(args):

    # MARK: TRAINING PREPARATION AND MODULES

    # Initialize all the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Set the output format to print into the console and save into LOG file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + ".log")
        ]
    )

    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Construct the model
    slrt_model = SPOTER(num_classes=args.num_classes, hidden_dim=args.hidden_dim)
    slrt_model.train(True)
    slrt_model.to(device)

    #  Print the model summary
    # Print number of trainable parameters

    logging.info(slrt_model)
    # print(slrt_model)

    total_params = sum(p.numel() for p in slrt_model.parameters() if p.requires_grad)
    logging.info("Total number of trainable parameters: {}".format(total_params))
    print("Total number of trainable parameters: {}".format(total_params))
    # Construct the other modules
    cel_criterion = nn.CrossEntropyLoss()
    sgd_optimizer = optim.SGD(slrt_model.parameters(), lr=args.lr, momentum = args.momentum, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, factor=args.scheduler_factor, patience=args.scheduler_patience)

    # Ensure that the path for checkpointing and for images both exist
    Path("out-checkpoints/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    Path("out-img/").mkdir(parents=True, exist_ok=True)


    # MARK: DATA

    # Training set
    transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    train_set = CzechSLRDataset(args.training_set_path, transform=transform, augmentations=True)

    # Validation set
    if args.validation_set == "from-file":
        val_set = CzechSLRDataset(args.validation_set_path)
        val_loader = DataLoader(val_set, shuffle=True, generator=g)

    elif args.validation_set == "split-from-train":
        train_set, val_set = __balance_val_split(train_set, 0.2)

        val_set.transform = None
        val_set.augmentations = False
        val_loader = DataLoader(val_set, shuffle=True, generator=g)

    else:
        val_loader = None

    # Testing set
    if args.testing_set_path:
        eval_set = CzechSLRDataset(args.testing_set_path)
        eval_loader = DataLoader(eval_set, shuffle=True, generator=g)

    else:
        eval_loader = None

    # Final training set refinements
    if args.experimental_train_split:
        train_set = __split_of_train_sequence(train_set, args.experimental_train_split)

    train_loader = DataLoader(train_set, shuffle=True, generator=g)


    # MARK: TRAINING
    train_acc, val_acc = 0, 0
    losses, train_accs, val_accs = [], [], []
    lr_progress = []
    top_train_acc, top_val_acc = 0, 0
    checkpoint_index = 0

    if args.experimental_train_split:
        print("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
        logging.info("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")

    else:
        print("Starting " + args.experiment_name + "...\n\n")
        logging.info("Starting " + args.experiment_name + "...\n\n")

    for epoch in range(args.epochs):
        train_loss, _, _, train_acc = train_epoch(slrt_model, train_loader, cel_criterion, sgd_optimizer, device)
        losses.append(train_loss.item() / len(train_loader))
        train_accs.append(train_acc)

        if val_loader:
            slrt_model.train(False)
            _, _, val_acc = evaluate(slrt_model, val_loader, device)
            slrt_model.train(True)
            val_accs.append(val_acc)

        # Save checkpoints if they are best in the current subset
        if args.save_checkpoints:
            if train_acc > top_train_acc:
                top_train_acc = train_acc
                torch.save(slrt_model, "out-checkpoints/" + args.experiment_name + "/checkpoint_t_" + str(checkpoint_index) + ".pth")

            if val_acc > top_val_acc:
                top_val_acc = val_acc
                torch.save(slrt_model, "out-checkpoints/" + args.experiment_name + "/checkpoint_v_" + str(checkpoint_index) + ".pth")

        if epoch % args.log_freq == 0:
            print("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item() / len(train_loader)) + " acc: " + str(train_acc))
            logging.info("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item() / len(train_loader)) + " acc: " + str(train_acc))

            if val_loader:
                print("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))
                logging.info("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))

            print("")
            logging.info("")

        # Reset the top accuracies on static subsets
        if epoch % 10 == 0:
            top_train_acc, top_val_acc = 0, 0
            checkpoint_index += 1

        lr_progress.append(sgd_optimizer.param_groups[0]["lr"])

    # MARK: TESTING

    print("\nTesting checkpointed models starting...\n")
    logging.info("\nTesting checkpointed models starting...\n")

    top_result, top_result_name, top_loss, top_train_acc = 0, "", 0, 0
    if eval_loader:
        for i in range(checkpoint_index):
            for checkpoint_id in ["t", "v"]:
                # tested_model = VisionTransformer(dim=2, mlp_dim=108, num_classes=100, depth=12, heads=8)
                tested_model = torch.load("out-checkpoints/" + args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i) + ".pth")
                tested_model.train(False)
                _, _, eval_acc = evaluate(tested_model, eval_loader, device, print_stats=True)

                if eval_acc > top_result:
                    top_result = eval_acc
                    top_result_name = args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i)
            
                print("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))
                logging.info("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))

        print("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
        logging.info("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
    # Append Accuracies with momentum and weight_decay in a csv file
    if args.save_checkpoints:
        with open("out-checkpoints" + "/results.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow([args.momentum, args.weight_decay, top_result, top_loss, top_train_acc])

    # PLOT 0: Performance (loss, accuracies) chart plotting
    if args.plot_stats:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(losses) + 1), losses, c="#D64436", label="Training loss")
        ax.plot(range(1, len(train_accs) + 1), train_accs, c="#00B09B", label="Training accuracy")

        if val_loader:
            ax.plot(range(1, len(val_accs) + 1), val_accs, c="#E0A938", label="Validation accuracy")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
        ax.grid()

        fig.savefig("out-img/" + args.experiment_name + "_loss.png")

    # PLOT 1: Learning rate progress
    if args.plot_lr:
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
        ax1.set(xlabel="Epoch", ylabel="LR", title="")
        ax1.grid()

        fig1.savefig("out-img/" + args.experiment_name + "_lr.png")

    print("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    logging.info("\nAny desired statistics have been plotted.\nThe experiment is finished.")

def find_best_hyperparameters(args):
    # MARK: HYPERPARAMETER SEARCH
    momentum_values =[0.99, 0.97, 0.95, 0.9, 0.7, 0.5, 0.2, 0.1, 0.01, 0]
    weight_decay_values = [1e-3, 1e-4, 1e-5, 0]
    lr = 2e-3

    best_result, best_momentum, best_weight_decay = 0, 0, 0

    # for momentum in momentum_values:
    momentum = 0
    for weight_decay in weight_decay_values:
        args.momentum = momentum
        args.weight_decay = weight_decay
        args.lr = lr
        args.experiment_name = "hyperparameter_search_" + str(momentum) + "_" + str(weight_decay)
        
        print("Momentum: " + str(momentum) + " Weight decay: " + str(weight_decay))
        logging.info("Momentum: " + str(momentum) + " Weight decay: " + str(weight_decay))

        # Train the model
        train(args)
        

def eval_hyperparam_checkpointed_models(args):
    momentum_values =[0.99, 0.97, 0.95, 0.9, 0.7, 0.5, 0.2, 0.1, 0.01, 0]
    weight_decay_values = [1e-3, 1e-4, 1e-5, 0]
    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    eval_set = CzechSLRDataset(args.testing_set_path)
    g = torch.Generator()
    g.manual_seed(args.seed)
    eval_loader = DataLoader(eval_set, shuffle=True, generator=g)
    for momentum in [0]:
        for weight_decay in [0]:
            # Load every checkpoint and evaluate it
            top_result, top_result_name, top_loss, top_train_acc = 0, "", 0, 0
            for i in range(3):
                for checkpoint_id in ["t", "v"]:
                    # tested_model = torch.load("out-checkpoints/hyperparameter_search_" + str(momentum) + "_" + str(weight_decay) + "/checkpoint_" + checkpoint_id + "_" + str(i) + ".pth")
                    tested_model = torch.load("out-checkpoints/baseline_model/checkpoint_" + checkpoint_id + "_" + str(i) + ".pth")
                    tested_model.train(False)
                    _, _, eval_acc = evaluate(tested_model, eval_loader, device, print_stats=True)

                    if eval_acc > top_result:
                        top_result = eval_acc
                        top_result_name = "hyperparameter_search_" + str(momentum) + "_" + str(weight_decay) + "/checkpoint_" + checkpoint_id + "_" + str(i)
                    
                    print("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))
            
            # Append Accuracies with momentum and weight_decay in a csv file
            with open("out-checkpoints" + "/results.csv", "a") as f:
                f.write(str(momentum) + "," + str(weight_decay) + "," + str(top_result) + "\n")

def finetune_hyperparam_checkpointed_models(args):
    # Initialize all the seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    l = [(0.2, 1e-3), (0.1, 1e-3), (0.1, 1e-4), (0.1, 1e-5), (0.1, 0), (.01, 1e-3), (.01, 1e-4), (.01, 1e-5)]
    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    g = torch.Generator()
    g.manual_seed(args.seed)
    # MARK: DATA

    # Training set
    transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    train_set = CzechSLRDataset(args.training_set_path, transform=transform, augmentations=True)

    # Validation set
    if args.validation_set == "from-file":
        val_set = CzechSLRDataset(args.validation_set_path)
        val_loader = DataLoader(val_set, shuffle=True, generator=g)

    elif args.validation_set == "split-from-train":
        train_set, val_set = __balance_val_split(train_set, 0.2)

        val_set.transform = None
        val_set.augmentations = False
        val_loader = DataLoader(val_set, shuffle=True, generator=g)

    else:
        val_loader = None

    # Testing set
    if args.testing_set_path:
        eval_set = CzechSLRDataset(args.testing_set_path)
        eval_loader = DataLoader(eval_set, shuffle=True, generator=g)

    else:
        eval_loader = None

    # Final training set refinements
    if args.experimental_train_split:
        train_set = __split_of_train_sequence(train_set, args.experimental_train_split)

    train_loader = DataLoader(train_set, shuffle=True, generator=g)

    for momentum, weight_decay in l:
        # Set the output format to print into the console and save into LOG file
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("out-checkpoints/hyperparameter_search_" + str(momentum) + "_" + str(weight_decay) + "/LOG")
            ]
        )
        # load last checkpoint
        tested_model = torch.load("out-checkpoints/hyperparameter_search_" + str(momentum) + "_" + str(weight_decay) + "/checkpoint_t_2.pth")
        tested_model.to(device) 
        tested_model.train(True)

        cel_criterion = nn.CrossEntropyLoss()
        sgd_optimizer = optim.SGD(tested_model.parameters(), lr=2e-3, momentum = momentum, weight_decay = weight_decay)
        train_acc, val_acc = 0, 0
        losses, train_accs, val_accs = [], [], []
        lr_progress = []
        top_train_acc, top_val_acc = 0, 0
        checkpoint_index = 2
        # train for 30 epochs
        print("Starting training for momentum = " + str(momentum) + " and weight_decay = " + str(weight_decay))
        logging.info("Starting training for momentum = " + str(momentum) + " and weight_decay = " + str(weight_decay))
        for epoch in range(30):
            train_loss, _, _, train_acc = train_epoch(tested_model, train_loader, cel_criterion, sgd_optimizer, device)
            losses.append(train_loss)
            train_accs.append(train_acc)

            if val_loader:
                tested_model.train(False)
                _, _, val_acc = evaluate(tested_model, val_loader, device)
                val_accs.append(val_acc)
                tested_model.train(True)
            
            if args.save_checkpoints:
                if val_acc > top_val_acc:
                    top_val_acc = val_acc
                    torch.save(tested_model, "out-checkpoints/hyperparameter_search_" + str(momentum) + "_" + str(weight_decay) + "/checkpoint_t_" + str(checkpoint_index) + ".pth")
            
            if epoch % 5 == 0:
                print("Epoch " + str(epoch) + " train loss: " + str(train_loss) + " train acc: " + str(train_acc) + " val acc: " + str(val_acc))
                logging.info("Epoch " + str(epoch) + " train loss: " + str(train_loss) + " train acc: " + str(train_acc) + " val acc: " + str(val_acc))

                if val_loader:
                    print("Epoch " + str(epoch) + " val acc: " + str(val_acc))
                    logging.info("Epoch " + str(epoch) + " val acc: " + str(val_acc))
                
            if epoch % 10 == 0:
                checkpoint_index += 1
                top_train_acc = 0
                top_val_acc = 0

            lr_progress.append(sgd_optimizer.param_groups[0]['lr'])

        top_result, top_result_name, top_loss, top_train_acc = 0, "", 0, 0
        if eval_loader:
            for i in range(checkpoint_index):
                for checkpoint_id in ["t", "v"]:
                    # tested_model = VisionTransformer(dim=2, mlp_dim=108, num_classes=100, depth=12, heads=8)
                    tested_model = torch.load("out-checkpoints/hyperparameter_search_" + str(momentum) + "_" + str(weight_decay) + "/checkpoint_" + checkpoint_id + "_" + str(i) + ".pth")
                    tested_model.train(False)
                    _, _, eval_acc = evaluate(tested_model, eval_loader, device, print_stats=True)

                    if eval_acc > top_result:
                        top_result = eval_acc
                        top_result_name = "checkpoint_" + checkpoint_id + "_" + str(i)
                
                    print("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))
                    logging.info("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))

            print("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
            logging.info("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
        # Save the results
        with open("out-checkpoints/results_finetuned.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([momentum, weight_decay, val_acc])
        
        # Plot the results
        fig, ax = plt.subplots()
        ax.plot(range(1, len(losses) + 1), losses, c="#D64436", label="Training loss")
        ax.plot(range(1, len(train_accs) + 1), train_accs, c="#00B09B", label="Training accuracy")

        if val_loader:
            ax.plot(range(1, len(val_accs) + 1), val_accs, c="#E0A938", label="Validation accuracy")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
        ax.grid()

        fig.savefig("out-img/hyperparameter_finetuning_" + str(momentum) + "_" + str(weight_decay) + "_loss_acc.png")
        

def plot():    
    # Plot 3D surface of the results
    with open("Plot Input.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        momentum = []
        weight_decay = []
        accuracy = []
        for row in data:
            momentum.append(float(row[0]))
            weight_decay.append(float(row[1]))
            accuracy.append(float(row[2]))

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(momentum, weight_decay, accuracy, cmap='viridis', edgecolor='none')
        ax.set_title('Hyperparameter search results')
        ax.set_xlabel('Momentum')
        ax.set_ylabel('Weight decay')
        ax.set_zlabel('Accuracy')
        plt.show()

        fig.savefig("out-img/hyperparameter_search_results.png")

import seaborn as sns   
import pandas as pd
def confusion(args):
    # Plot confusion matrix
    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    eval_set = CzechSLRDataset(args.testing_set_path)
    g = torch.Generator()
    g.manual_seed(42)
    eval_loader = DataLoader(eval_set, shuffle=True, generator=g)
    tested_model = torch.load("out-checkpoints/hyperparameter_search_0.1_0.0001/checkpoint_t_2.pth")
    tested_model.train(False)
    _, _, eval_acc = evaluate(tested_model, eval_loader, device, print_stats=True)

    # Plot confusion matrix
    y_pred = []
    y_true = []
    for i, (x, y) in enumerate(eval_loader):
        x = x.squeeze(0).to(device)
        y = y.to(device, dtype=torch.long)

        out = tested_model(x).expand(1, -1, -1)

        # Get the predicted class
        m = int(torch.argmax(torch.nn.functional.softmax(out, dim=2)))
        y_pred.append(m)
        y_true.append(y.item())

    # Dataframe with true and predicted labels
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    # Save dataframe to csv
    df.to_csv("out-img/confusion_matrix_0.1_0.0001.csv", index=False)
    # Create confusion matrix
    confusion_matrix = pd.crosstab(df['y_true'], df['y_pred'], rownames=['Actual'], colnames=['Predicted'])
    # Plot heatmap
    fig = plt.figure()
    sns.heatmap(confusion_matrix, annot=True)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # Change dimensions of the plot
    fig.set_size_inches(10, 10)
    plt.show()

    fig.savefig("out-img/confusion_matrix_0.1_0.0001.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    # train_findlr(args)
    # train(args)
    # find_best_hyperparameters(args)
    eval_hyperparam_checkpointed_models(args)
    # plot()
    # confusion(args)
    # finetune_hyperparam_checkpointed_models(args)