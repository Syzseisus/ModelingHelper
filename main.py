import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict as ddict

import torch
import torch.nn as nn
import torch.optim as optim

# import python files
from models.model import MyModel
from utils.metric import accuracy, mse
from utils.utils import set_random_seed, parameter_parser, Loader

# setting scheduler
from torch.optim.lr_scheduler import StepLR, LambdaLR, CosineAnnealingLR

scheduler_dict = {
    'Step': StepLR,
    'Lambda': LambdaLR,
    'Cosine': CosineAnnealingLR,
}


class MyProject(object):
    def __init__(self, train_loader, valid_loader, test_loader, args):
        '''
        Order of Functions
            1. set_scheduler
            2. load_pre_trained_weights
            3. train
            4. eval_
            5. run
        '''
        self.args = args
        self.device = args.device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.log_dir = args.log_dir
        self.results = ddict(list)

        # fix seed
        set_random_seed(args.seed)

        # region: Init Model
        with Loader("Initialize Model...", end="Model Initialized.", steps="\\ | / -".split()):
            self.model = MyModel().to(self.device)  # as you want
            if args.load_pretrain:
                self.load_pre_trained_weights()
            import time
            time.sleep(5)
        print(f"{' MODEL SPEC ':=^80}")
        print(self.model)
        print("=" * 80)
        # endregion

        # region: loss, optimizer, scheduler
        self.criterion = nn.NLLLoss()  # as you want
        self.optimizer = optim.SGD(  # as you want
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd,
        )
        if args.scheduler:
            self.scheduler = self.set_scheduler()
        # endregion

        print("Done")

    def set_scheduler(self):
        if self.args.scheduler == 'Step':
            self.scheduler = StepLR(
                self.optimizer, step_size=self.args.steplr_step_size, gamma=self.args.steplr_gamma,
            )
        elif self.args.scheduler == 'Lambda':
            self.scheduler = LambdaLR(
                self.optimizer, lr_lambda=self.args.lambdalr_lambda,
            )
        elif self.args.scheduler == 'Cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.args.cosinelr_period,
            )
        else:
            raise NotImplementedError()

    def load_pre_trained_weights(self):
        try:
            state_dict_path = os.path.join(self.args.pretrain_model_path, 'model.pt')
            self.model.load_state_dict(torch.load(state_dict_path))
            print(f"\n\tLoaded pre-trained model from {state_dict_path} with success.")
        except FileNotFoundError:
            print(f"There isn't {state_dict_path}. Training from scratch.")

    def train(self, epoch, counter):
        self.model.train()

        train_loss = 0.0
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"  Train - Counter: {counter}/{self.args.patience}",
        )
        for bn, (X, y) in pbar:
            pbar.set_description(f"  Train - Counter: {counter}/{self.args.patience}")

            self.optimizer.zero_grad()

            X = X.to(self.device)
            y = y.to(self.device)
            logit = self.model(X)
            loss = self.criterion(logit, y)

            loss.backward()
            self.optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            train_loss += loss.item()

        train_loss /= self.args.num_train_data
        self.results['train_loss'].append(train_loss)

        if self.args.scheduler and epoch >= self.args.warmup:
            self.scheduler.step()

    def eval_(self, epoch=0, counter=0, mode='valid'):
        loader = self.valid_loader if mode == 'valid' else self.test_loader
        desc = f"  Valid - Counter: {counter}/{self.args.patience}" if mode == 'valid' else '  Test '

        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=desc,
        )
        with torch.no_grad():
            self.model.eval()

            total_loss = 0.0
            total_acc = 0.0
            total_mse = 0.0
            for bn, (X, y) in pbar:
                desc = f"  Valid - Counter: {counter}/{self.args.patience}" if mode == 'valid' else '  Test '
                pbar.set_description(desc)

                self.model.train()
                self.optimizer.zero_grad()

                X = X.to(self.device)
                y = y.to(self.device)
                logit = self.model(X)
                loss = self.criterion(logit, y)
                acc_ = accuracy(logit, y)
                mse_ = mse(logit, y)

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", acc=f"{acc_:.4f}", mse=f"{mse_:.4f}"
                )
                total_acc += acc_
                total_mse += mse_
                total_loss += loss.item()

            total_loss /= self.args.num_valid_data if mode == 'valid' else self.args.num_test_data
            total_acc /= self.args.num_valid_data if mode == 'valid' else self.args.num_test_data
            total_mse /= self.args.num_valid_data if mode == 'valid' else self.args.num_test_data
            self.results['valid_loss'].append(total_acc)
            self.results['valid_acc'].append(total_acc)
            self.results['valid_mse'].append(total_mse)

        return total_loss, total_acc, None

    def run(self):
        print()
        counter = 0
        best_valid_acc = 0.0
        best_valid_mse = float('inf')
        digit_total = len(str(self.args.epochs))
        for epoch in range(self.args.epochs):
            digit_epoch = len(str(epoch))
            epochprint = ' ' * (digit_total - digit_epoch) + str(epoch + 1)
            print(f"Epoch: {epochprint}/{self.args.epochs}")

            self.train(epoch, counter)

            valid_loss, valid_acc, valid_mse = self.eval_(epoch, counter, mode='valid')
            # choose one condition. BE CAREFUL OF THE ORDER
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'model.pt'))
                counter = 0
            elif valid_mse < best_valid_mse:
                best_valid_mse = valid_mse
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'model.pt'))
                counter = 0
            else:
                counter += 1
                if counter == self.args.patience:
                    print(f"{' EARLY STOPPED! ':=^80}")
                    break

            if (epoch + 1) % self.args.safety_save:
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, f'model_{epoch + 1}.pt'))

        _, Accuracy, MSE = self.eval_(mode='test')
        print(f"Accuracy: {Accuracy:.2f}, MSE: {MSE:.4f}")


def main():
    # region: set and print arguments
    args = parameter_parser()

    print(' ' * 9, f"┏{'  ARGS  ':━^60}┓")
    for hyperparam in sorted(vars(args).keys()):
        val = f"{vars(args)[hyperparam]}"
        print(' ' * 9, f"┃ --{hyperparam:<25} {val:<30} ┃")
    print(' ' * 9, f"┗{'':━^60}┛\n")
    # endregion

    # region: Load Dataset
    with Loader("Load Dataset...", end="Dataset Loaded.", steps="\\ | / -".split()):
        train_set, valid_set, test_set =  # dataset
        train_loader, valid_loader, test_loader =  # dataset_to_dataloader

        args.num_train_data = len(train_set)
        args.num_valid_data = len(valid_set)
        args.num_test_data = len(test_set)
    # endregion

    # region: Main
    print("\nReady for the Project\n")
    myproject = MyProject(train_loader, valid_loader, test_loader, args)

    myproject.run()
    # endregion

    # region: saving
    print("\nProject is done.")
    with Loader("Save Results...", end="Results Saved.", steps="\\ | / -".split()):
        with open(os.path.join(args.log_dir, 'results.pk'), 'w') as f:
            json.dump(myproject.results, f)

        # plotting
        plt.plot(myproject.results['train_loss'])
        plt.plot(myproject.results['valid_loss'])
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(['train', 'valid'])
        plt.savefig(os.path.join(args.log_dir, 'loss_curve.png'))
        plt.close()

        plt.plot(myproject.results['valid_acc'])
        plt.title("Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(args.log_dir, 'val_accuracy.png'))
        plt.close()
    # endregion


if __name__ == "__main__":
    main()
