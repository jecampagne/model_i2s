import argparse
import json
import os
import random
import pathlib
import pickle
from types import SimpleNamespace
import multiprocessing
import time

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import regex as re
import glob


from model import *

# vesion 16 april 25: dataset single file for image+spec


################
# Utils
################
def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


################
# dataloader
################

# bug JEC 9 avril 25
# def get_list(dir, pattern):
#    dir = pathlib.Path(dir)
#    return list(dir.glob(pattern))

file_pattern = re.compile(r".*?(\d+).*?")


def get_order(file):
    match = file_pattern.match(os.path.basename(file))
    if not match:
        return math.inf
    return int(match.groups()[-1])


def get_list(dir, pattern):
    # dir = pathlib.Path(dir)
    a = list(glob.glob(dir + "/" + pattern))
    return sorted(a, key=get_order)


class CustumDataset(Dataset):
    """Load the data set which is supposed to be a Numpy structured array
    'img': the images tensors  N H W
    'spec': the original spectrum
    """

    def __init__(self, data_path, *, transform=None, ndata=None):

        if ndata is None:
            self.datas = get_list(data_path, "d*.npz")
        else:
            self.datas = get_list(data_path, "d*.npz")[:ndata]
        
        print(f"CustumDataset: {len(self.datas)} data loaded")

        self.transform = transform

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = np.load(self.datas[index])
        image = data["imag"]  # HxW
        image = np.expand_dims(image, axis=0)  # 1xHxW
        clean = data["imag_clean"]  # HxW
        clean = np.expand_dims(clean, axis=0)  # 1xHxW

        # to torch tensor
        image = torch.from_numpy(image)
        clean = torch.from_numpy(clean)

        # choice to transform as torch array
        # overwise make it as numpy array
        if self.transform is not None:
            image = self.transform(image)

        return image, clean


################
# train/test 1-epoch
################
def train(args, model, criterion, train_loader, optimizer, epoch):
    model.train()
    loss_sum = 0  # to get the mean loss over the dataset
    for i_batch, sample_batched in enumerate(train_loader):
        # get the "X,y"
        imgs = sample_batched[0].to(args.device)
        cleans = sample_batched[1].to(args.device)

        # train step
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, cleans)
        loss_sum += loss.item()
        # backprop to compute the gradients
        loss.backward()
        # perform an optimizer step to modify the weights
        optimizer.step()

    return loss_sum / (i_batch + 1)


def test(args, model, criterion, test_loader, epoch):
    model.eval()
    loss_sum = 0  # to get the mean loss over the dataset
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            # get the "X,y"
            imgs = sample_batched[0].to(args.device)
            cleans = sample_batched[1].to(args.device)
            #
            output = model(imgs)
            loss = criterion(output, cleans)
            loss_sum += loss.item()

    return loss_sum / (i_batch + 1)


################
# Main: init & loop on epochs
################


def main():

    # Training config
    parser = argparse.ArgumentParser(description="Image-to-Spectre model (I2SM)")
    parser.add_argument("--file", help="Config file")
    args0 = parser.parse_args()

    with open(args0.file) as jsonfile:
        settings_dict = json.load(jsonfile)
    args = SimpleNamespace(**settings_dict)

    # check number of num_workers
    NUM_CORES = multiprocessing.cpu_count()
    if args.num_workers >= NUM_CORES:
        print("Info: # workers set to", NUM_CORES // 2)
        args.num_workers = NUM_CORES // 2

    # where to find the dataset (train & test) (input images & output spectra)
    args.train_dir = args.data_root_path + args.train_dir
    args.test_dir = args.data_root_path + args.test_dir

    # where to put all model training stuff
    args.out_root_dir = args.out_root_dir + "/" + args.run_tag + "/"

    try:
        os.makedirs(
            args.out_root_dir, exist_ok=False
        )  # avoid erase the existing directories (exit_ok=False)
    except OSError:
        pass

    print("Info: outdir is ", args.out_root_dir)

    # device cpu/gpu...
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # seeding
    set_seed(args.seed)

    # dataset & dataloader
    ds_train = CustumDataset(args.train_dir,ndata=100_000)
    ds_test = CustumDataset(args.test_dir)#,ndata=500)

    train_loader = DataLoader(
        dataset=ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=ds_test,
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # get a batch to determin the image/spectrum sizes
    train_img, train_clean = next(iter(train_loader))
    # img_channels = args.num_channels
    img_H = train_img.shape[2]
    img_W = train_img.shape[3]
    print("image sizes: HxW", img_H, img_W)

    # model instantiation
    if args.archi == "Unet-Full":
        model = UNet(args)
    else:
        print("Error: ", args.archi, "unknown")
        return

    # check ouptut of model is ok. Allow to determine the model config done at run tile
    out = model(train_img)
    assert out.shape == train_clean.shape
    print(
        "number of parameters is ",
        sum(p.numel() for p in model.parameters() if p.requires_grad) // 10**6,
        "millions",
    )
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # put model to device before loading scheduler/optimizer parameters
    model.to(args.device)

    # optimizer & scheduler

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_init,
        # amsgrad=True,
        # eps=1e-8, # by default is 1e-8
        # weight_decay=1e-3   # default is 0
    )

    if args.use_scheduler:
    
        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs,                                                     eta_min=1e-5)
        elif args.scheduler == "reduce":
            scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=args.lr_decay, patience=args.patience, min_lr=1e-5)        
        else:
            print("FATAL: not known scheduler...")
            return


            

    # check for resume session: load model/optim/scheduler dictionnaries
    start_epoch = 0

    train_loss_history = []
    test_loss_history = []

    if args.resume:
        args.checkpoint_file = args.out_root_dir + args.checkpoint_file
        args.history_loss_cpt_file = args.out_root_dir + args.history_loss_cpt_file

        # load checkpoint of model/scheduler/optimizer
        if os.path.isfile(args.checkpoint_file):
            print("=> loading checkpoint '{}'".format(args.checkpoint_file))
            checkpoint = torch.load(args.checkpoint_file)
            # the first epoch for the new training
            start_epoch = checkpoint["epoch"]
            # model update state
            model.load_state_dict(checkpoint["model_state_dict"])
            if args.resume_scheduler:
                # optizimer update state
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # scheduler update state
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                print("=>>> scheduler not resumed")
                if args.resume_optimizer:
                    # optizimer update state
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                else:
                    print("=>>> optimizer not resumed")
            print("=> loaded checkpoint")
        else:
            print("=> FATAL no  checkpoint '{}'".format(args.checkpoint_file))
            return

        # load previous history of losses
        if os.path.isfile(args.history_loss_cpt_file):
            loss_history = np.load(args.history_loss_cpt_file, allow_pickle=True)
            train_loss_history = loss_history[0].tolist()
            test_loss_history = loss_history[1].tolist()

        else:
            print(
                "=> FATAL no history loss checkpoint '{}'".format(
                    args.history_loss_cpt_file
                )
            )
            return

    else:
        print("=> no checkpoints then Go as fresh start")

    # loss
    #criterion = nn.MSELoss(reduction="mean")
    criterion = nn.L1Loss(reduction="mean")

    # loop on epochs
    t0 = time.time()
    best_test_loss = np.inf

    print("The current args:", args)

    for epoch in range(start_epoch, args.num_epochs + 1):
        # training
        train_loss = train(args, model, criterion, train_loader, optimizer, epoch)
        # test
        test_loss = test(args, model, criterion, test_loader, epoch)

        # print & book keeping
        print(
            f"Epoch {epoch}, Losses train: {train_loss:.6f}",
            f"test {test_loss:.6f}, LR= {scheduler.get_last_lr()}",
            f"time {time.time()-t0:.2f}"
        )
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        # update scheduler
        if args.use_scheduler:
            if args.scheduler == "reduce":
                # Warning ReduceLROnPlateau needs a metric
                scheduler.step(test_loss)
            else:
                scheduler.step()

        # save state at each epoch to be able to reload and continue the optimization
        if args.use_scheduler:
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
        else:
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

        torch.save(
            state,
            args.out_root_dir + "/" + args.archi + "_last_state.pth",
        )
        # save intermediate history
        np.save(
            args.out_root_dir + "/" + args.archi + "_last_history.npy",
            np.array((train_loss_history, test_loss_history)),
        )

        # if better loss update best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        torch.save(
            state,
            args.out_root_dir + "/" + args.archi + "_best_state.pth",
        )

    # Bye
    tf = time.time()
    print("all done!", tf - t0)


################################
if __name__ == "__main__":
    main()
