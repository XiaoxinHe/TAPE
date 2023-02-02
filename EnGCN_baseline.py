import gc
import json
import os
import random
from datetime import datetime

import numpy as np
import torch

from options.base_options import BaseOptions
from trainer import trainer
from utils import print_args


def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


def main(args):
    list_test_acc = []
    list_valid_acc = []
    list_train_loss = []

    filedir = f"./logs/{args.dataset}"
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    if not args.exp_name:
        filename = f"{args.type_model}.json"
    else:
        filename = f"{args.exp_name}.json"
    path_json = os.path.join(filedir, filename)

    try:
        resume_seed = 0
        if os.path.exists(path_json):
            if args.resume:
                with open(path_json, "r") as f:
                    saved = json.load(f)
                    resume_seed = saved["seed"] + 1
                    list_test_acc = saved["test_acc"]
                    list_valid_acc = saved["val_acc"]
                    list_train_loss = saved["train_loss"]
            else:
                t = os.path.getmtime(path_json)
                tstr = datetime.fromtimestamp(t).strftime("%Y_%m_%d_%H_%M_%S")
                os.rename(
                    path_json, os.path.join(
                        filedir, filename + "_" + tstr + ".json")
                )
        if resume_seed >= args.N_exp:
            print("Training already finished!")
            return
    except:
        pass

    print_args(args)

    if args.debug_mem_speed:
        trnr = trainer(args)
        trnr.mem_speed_bench()

    for seed in range(resume_seed, args.N_exp):
        if args.seed is not None:
            seed = args.seed
        else:
            pass
        print(f"seed (which_run) = <{seed}>")

        args.random_seed = seed
        set_seed(args)
        # torch.cuda.empty_cache()
        trnr = trainer(args)
        if args.type_model in [
            "SAdaGCN",
            "AdaGCN",
            "GBGCN",
            "AdaGCN_CandS",
            "AdaGCN_SLE",
            "EnGCN",
        ]:
            train_loss, valid_acc, test_acc = trnr.train_ensembling(seed)
        else:
            train_loss, valid_acc, test_acc = trnr.train_and_test(seed)
        list_test_acc.append(test_acc)
        list_valid_acc.append(valid_acc)
        list_train_loss.append(train_loss)

        del trnr
        torch.cuda.empty_cache()
        gc.collect()

        # record training data
        print(
            "mean and std of test acc: {:.4f} {:.4f} ".format(
                np.mean(list_test_acc) * 100, np.std(list_test_acc) * 100
            )
        )

        try:
            to_save = dict(
                seed=seed,
                test_acc=list_test_acc,
                val_acc=list_valid_acc,
                train_loss=list_train_loss,
                mean_test_acc=np.mean(list_test_acc),
                std_test_acc=np.std(list_test_acc),
            )
            with open(path_json, "w") as f:
                json.dump(to_save, f)
        except:
            pass
    print(
        "final mean and std of test acc: ",
        f"{np.mean(list_test_acc)*100:.4f} $\\pm$ {np.std(list_test_acc)*100:.4f}",
        "final mean and std of val acc: ",
        f"{np.mean(list_valid_acc) * 100:.4f} $\\pm$ {np.std(list_valid_acc) * 100:.4f}",
    )


if __name__ == "__main__":
    args = BaseOptions().initialize()
    main(args)
