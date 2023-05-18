import os
import torch
import random
import argparse
import numpy as np
from time import sleep
from itertools import cycle
from threading import Thread
from datetime import datetime
from shutil import get_terminal_size
import torch.backends.cudnn as cudnn

possible_scheduler = [
    'Step',
    'Lambda',
    'Cosine',
]


class Loader:
    '''
    Thanks to @ted, https://stackoverflow.com/questions/22029562/python-how-to-make-simple-animated-loading-while-process-is-running
    A loader-like context manager

    Args:
        desc     (str, optional)   : The loader's description. Defaults to "Loading..."
        end      (str, optional)   : Final print. Defaults to "Done!".
        steps    (list, optional)  : List of strings for animating. Defaults to ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        steptime (float, optional) : Sleep time between prints. Defaults to 0.1s.
    '''
    def __init__(self, desc="Loading...", end="Done!", steps=["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"], steptime=0.1):
        self.desc = desc
        self.end = end
        self.steps = steps
        self.steptime = steptime

        self._thread = Thread(target=self._animate, daemon=True)
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c}", flush=True, end="")
            sleep(self.steptime)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end='', flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_teyp, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()


def set_random_seed(seed=0, determ=False):
    """
    set random seed.
    :param seed: int, random seed to use
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = determ  # True for experiment precisely


def parameter_parser():
    description = ""
    parser = argparse.ArgumentParser(description=description)

    # region: experiment configuration
    parser.add_argument(
        "--seed",
        type=int,
        default=1013,
        help="Random seed for train-test split. Default is 1013.",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="The number of GPU gonna use. \
            -1 for cpu, for multi-gpu, provide the GPU numbers as '13' and turn on args.multi_gpu",
    )

    parser.add_argument(
        "--multi_gpu",
        action='store_true',
        help="Whether use multi-gpu",
    )

    parser.add_argument(
        "--safety_save",
        type=int,
        default=-1,
        help="Number of training rounds at which to save. Default is -1 for not saving.",
    )
    # endregion

    # region: model training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of training epochs. Default is 2000.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Number of training rounds before early stopping. Default is 100.\
            -1 for not using earlystopping.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch Size. Default is 128.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate. Default is 1e-4.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of training rounds before start scheduling. Default is 0.",
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default='',
        choices=[''] + possible_scheduler,
        help="Scheduler be used. Default is '', meaning No scheduler.",
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=0.005,
        help="Weight matrix regularization. Default is 0.005.",
    )

    parser.add_argument(
        "--load_pretrain",
        action='store_true',  # or store_false
        help="Whether load pretrain model. If it is on, pretrain_model_path is required.",
    )

    parser.add_argument(
        "--pretrain_model_path",
        type=str,
        default='',
        help="Where the dataset is stored"
    )
    # endregion

    # region: data configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default='MNIST',
        help="Name of dataset",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default='./root/MNIST',
        help="Where the dataset is stored"
    )
    # endregion

    # region: helper
    # parser.add_argument(
    #     "--numerical",
    #     type=int,  # or float
    #     default=10,
    #     help="numerical argument",
    # )

    # parser.add_argument(
    #     "--string",
    #     type=str,
    #     default='string',
    #     help="string argument",
    # )

    # parser.add_argument(
    #     "--boolean",
    #     action='store_true',  # or store_false
    #     help="boolean action argument",
    # )

    # parser.add_argument(
    #     "--list",
    #     nargs='+',
    #     required=True,
    #     type=int,  # type of elements in list
    #     help="list argument. set default with parser.set_defaults(list=[]).",
    # )
    # parser.set_defaults(list=[1, 2, 3])
    # endregion

    args = parser.parse_args()

    # set device
    if args.gpu >= 0 and torch.cuda.is_available():
        if args.multi_gpu:
            # Todo: multi gpu 설정하기
            pass
        else:
            args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    # check
    if args.load_pretrain:
        if not args.pretrain_model_path:
            print("Please provide pretrained model path!")

    # scheduler
    if args.scheduler:
        args = set_scheduler_params(args)

    # log_dir
    now = datetime.now().strftime("%m-%d-%H-%M-%S")
    args.log_dir = os.path.join('results', args.dataset, now)
    os.makedirs(args.log_dir, exist_ok=True)

    return args


def set_scheduler_params(args):
    if args.scheduler == 'Step':
        args.steplr_step_size = 10
        args.steplr_gamma = 0.5
    elif args.scheduler == 'Lambda':
        args.lambdalr_lambda = 0.95
    elif args.scheduler == 'Cosine':
        args.cosinelr_period = 50

    return args
