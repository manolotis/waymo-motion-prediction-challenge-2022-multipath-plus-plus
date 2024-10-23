import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from model.multipathpp import MultiPathPP
from model.data import get_dataloader, dict_to_cuda, normalize
from model.losses import pytorch_neg_multi_log_likelihood_batch, nll_with_covariances
import subprocess
import os
import glob
import random
from utils.train_utils import parse_arguments, get_config

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Waymo Values
# MEAN_X = 1.4715e+01
# MEAN_Y = 4.3008e-03
# STD_XY = 10.

# CARLA Behavior Agent values (in Town05)
MEAN_X = 20.424562
MEAN_Y = 0.0039684023
STD_XY = (20.241842 + 14.278944) / 2.0

def get_last_file(path):
    list_of_files = glob.glob(f'{path}/*')
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def get_best_checkpoint(path):
    list_of_files = glob.glob(f'{path}/*')
    for f in list_of_files:
        if "best" in f and "old" not in f:
            return f
    return None


def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


args = parse_arguments()
config = get_config(args)
try:
    trained_models_path = "../trained_models"
    if not os.path.exists(trained_models_path):
        os.mkdir(trained_models_path)
    models_path = os.path.join(trained_models_path, f"{config['config_name']}__{get_git_revision_short_hash()}")
    os.mkdir(models_path)
except FileExistsError:
    pass
except Exception as e:
    print("Could not make path")
    raise e

# last_checkpoint_path = get_last_file(models_path)
best_checkpoint_path = get_best_checkpoint(models_path)

dataloader = get_dataloader(config["train"]["data_config"])
val_dataloader = get_dataloader(config["val"]["data_config"])
model = MultiPathPP(config["model"])
model.cuda()
optimizer = Adam(model.parameters(), **config["train"]["optimizer"])
if config["train"]["scheduler"]:
    scheduler = ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)

last_epoch = 0
# if last_checkpoint_path is not None:
if best_checkpoint_path is not None:
    best_checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    # optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
    # num_steps = last_checkpoint["num_steps"]
    # last_epoch = last_checkpoint["epoch"]
    # train_losses = last_checkpoint["train_losses"]
    # val_losses = last_checkpoint["val_losses"]

    # if config["train"]["scheduler"]:
    #     scheduler.load_state_dict(last_checkpoint["scheduler_state_dict"])

    num_steps = 0
    train_losses = []
    val_losses = []


    # print("LOADED ", last_checkpoint_path)
    print("LOADED ", best_checkpoint_path)
    print("Epoch: ", last_epoch)
    print("num_steps: ", num_steps)
    print("len(train_losses): ", len(train_losses))
    print("len(val_losses): ", len(val_losses))

else:
    num_steps = 0
    train_losses = []
    val_losses = []
    print("Training from scratch")

this_num_steps = 0
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("N PARAMS=", params)

epochs_without_improvement = 0
for epoch in tqdm(range(last_epoch, config["train"]["n_epochs"])):
    pbar = tqdm(dataloader)
    epoch_losses_train = []
    epoch_losses_val = []

    for data in pbar:
        model.train()
        optimizer.zero_grad()
        if config["train"]["normalize"]:
            data = normalize(data, config)
        dict_to_cuda(data)
        probas, coordinates, covariance_matrices, loss_coeff = model(data, num_steps)
        assert torch.isfinite(coordinates).all()
        assert torch.isfinite(probas).all()
        assert torch.isfinite(covariance_matrices).all()
        xy_future_gt = data["target/future/xy"]
        if config["train"]["normalize_output"]:
            xy_future_gt = (data["target/future/xy"] - torch.Tensor([MEAN_X, MEAN_Y]).cuda()) / STD_XY
            assert torch.isfinite(xy_future_gt).all()
        loss = nll_with_covariances(
            xy_future_gt, coordinates, probas, data["target/future/valid"].squeeze(-1),
            covariance_matrices) * loss_coeff
        epoch_losses_train.append(loss.item())
        loss.backward()
        if "clip_grad_norm" in config["train"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["clip_grad_norm"])
        optimizer.step()

        # if config["train"]["normalize_output"]:
        #     _coordinates = coordinates.detach() * STD_XY + torch.Tensor([MEAN_X, MEAN_Y]).cuda()
        # else:
        #     _coordinates = coordinates.detach()
        if num_steps % 10 == 0:
            pbar.set_description(
                f"epoch = {epoch} | "
                f"epoch avg. loss = {np.mean(epoch_losses_train):.2} | "
                f"step loss = {round(loss.item(), 2)} | "
                f"lr: {optimizer.param_groups[0]['lr']:.3} | "
                f"step = {num_steps} | "
                f"ewi: {epochs_without_improvement}")
        # if num_steps % 1000 == 0 and this_num_steps > 0:
        if (num_steps + 1) % len(dataloader) == 0 and this_num_steps > 0:
            saving_data = {
                "epoch": epoch,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "num_steps": num_steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            if config["train"]["scheduler"]:
                saving_data["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(saving_data, os.path.join(models_path, f"last.pth"))
            if this_num_steps % 15 == 0:
                torch.save(saving_data, os.path.join(models_path, f"e{epoch}_it{num_steps}.pth"))
        # if num_steps % config["train"]["validate_every_n_steps"] == 0 and this_num_steps > 0:

        num_steps += 1
        this_num_steps += 1
        if "max_iterations" in config["train"] and num_steps > config["train"]["max_iterations"]:
            break

    del data
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        losses = []
        min_ades = []
        first_batch = True
        total_validation_loss = 0
        n_batches = 0

        pbar2 = tqdm(val_dataloader)
        for data in pbar2:
            # for data in val_dataloader:
            if config["train"]["normalize"]:
                data = normalize(data, config)
            dict_to_cuda(data)
            # probas, coordinates, _, _ = model(data, num_steps)
            probas, coordinates, covariance_matrices, loss_coeff = model(data, num_steps)

            # if config["train"]["normalize_output"]:
            #     coordinates = coordinates * STD_XY + torch.Tensor([MEAN_X, MEAN_Y]).cuda()

            if config["train"]["normalize_output"]:
                xy_future_gt = (data["target/future/xy"] - torch.Tensor([MEAN_X, MEAN_Y]).cuda()) / STD_XY
                assert torch.isfinite(xy_future_gt).all()

            loss = nll_with_covariances(
                xy_future_gt, coordinates, probas, data["target/future/valid"].squeeze(-1),
                covariance_matrices) * loss_coeff
            epoch_losses_val.append(loss.item())

            pbar2.set_description(
                f"epoch = {epoch} | "
                f"epoch avg. val loss= {np.mean(epoch_losses_val):.2} | "
                f"step loss = {round(loss.item(), 2)}")

    train_losses.append(np.mean(epoch_losses_train))
    val_losses.append(np.mean(epoch_losses_val))

    scheduler.step(val_losses[-1])

    saving_data = {
        "epoch": epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "num_steps": num_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if config["train"]["scheduler"]:
        saving_data["scheduler_state_dict"] = scheduler.state_dict()

    best_epoch = val_losses[-1] < np.min(val_losses[:-1]) if epoch > 0 else True

    epochs_without_improvement += 1

    if best_epoch:
        epochs_without_improvement = 0
        print("Best validation loss. Saving best model...\n")
        torch.save(saving_data, os.path.join(models_path, f"best.pth"))

    should_break = ("max_iterations" in config["train"] and num_steps > config["train"][
        "max_iterations"]) or epochs_without_improvement >= config["train"]["patience"]
    if should_break:
        print("Reached stop criteria.")
        break
