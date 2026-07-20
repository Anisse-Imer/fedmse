"""
Centralized training under data-poisoning attack — comparison baseline.

Mirrors the Byzantine FL experiment (byzantine_combined_main.py) but removes
the federation entirely: all client data is pooled into a single global dataset
and one model is trained directly.

The poisoning setup is kept identical to the FL experiment so results are
directly comparable:
  - Byzantine ratio : 30 % of clients (fixed)
  - Poisoning ratio : 30 % of a Byzantine client's training samples replaced
                      with abnormal traffic before merging into the global set
  - Runs            : 5 independent runs (Byzantine clients re-drawn each time)
  - Models          : Autoencoder and Hybrid (SAE)
  - Phases / rounds : 20 (each phase = 100 training epochs)
  - Early stopping  : patience = 3 epochs on validation loss

Note: model corruption is not applicable here — there is no weight exchange in
centralized training.  This script isolates the effect of data poisoning alone
in a centralized setting, so it can be compared against both the clean
centralized baseline (centralized_main.py) and the combined FL attack
(byzantine_combined_main.py).

Usage:
  python centralized_poisoned_main.py \
      --file Configuration/scen2-nba-iot-centralized-poisoned-30clients-iid.json
"""

import os
import json
import pickle
import argparse
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from Model import Shrink_Autoencoder, Autoencoder
from DataLoader import load_data, IoTDataset, IoTDataProccessor
from Evaluator import Evaluator

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", required=True,
                    help="Path to JSON config file (relative to src/).")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Hyper-parameters — identical to both centralized_main.py and
# byzantine_combined_main.py for a fair three-way comparison
# ---------------------------------------------------------------------------
epoch          = 100
num_phases     = 20
num_runs       = 5
lr_rate        = 1e-5
shrink_lambda  = 10
data_seed      = 1234
batch_size     = 12
dim_features   = 115
metric         = "AUC"
patience       = 3
scen_name      = "CentralizedPoisoned"

BYZANTINE_RATIO  = 0.3
POISONING_RATIO  = 0.3

config_file  = args.file
_cfg_stem    = os.path.splitext(os.path.basename(config_file))[0]
distribution = "nonIID" if "non-iid" in _cfg_stem.lower() else "IID"

with open(config_file) as f:
    _cfg = json.load(f)
network_size = len(_cfg["devices_list"])

byz_pct = int(BYZANTINE_RATIO * 100)
no_Exp  = (
    f"CentralizedPoisoned_{distribution}_byz{byz_pct}pct_"
    f"{epoch}epoch_{network_size}client_{num_phases}phases_"
    f"lr{lr_rate}_lamda{shrink_lambda}_dataseed{data_seed}"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def poison_train_slice(train_df, abnormal_arr, proc):
    """
    Replace POISONING_RATIO of a client's training rows with abnormal samples.
    Returns a numpy array in the same scaled space as proc_train.
    """
    proc_train, _ = proc.transform(train_df)
    n_total  = len(proc_train)
    n_poison = max(1, int(n_total * POISONING_RATIO))
    n_clean  = n_total - n_poison

    idx    = np.random.choice(len(abnormal_arr), n_poison,
                              replace=(n_poison > len(abnormal_arr)))
    return np.concatenate([proc_train[:n_clean], abnormal_arr[idx]], axis=0)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for batch in loader:
        _, _, loss = model(batch[0].to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total += loss.item()
    return total / len(loader)


def validate_epoch(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            _, _, loss = model(batch[0].to(device))
            total += loss.item()
    return total / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_seeds(data_seed)

    # ------------------------------------------------------------------ #
    # Load raw data once — reused across all runs                          #
    # ------------------------------------------------------------------ #
    with open(config_file) as f:
        config = json.load(f)

    raw_clients = []
    for device in config["devices_list"][:network_size]:
        normal_path   = os.path.join(config["data_path"], device["normal_data_path"])
        abnormal_path = os.path.join(config["data_path"], device["abnormal_data_path"])
        test_np_path  = os.path.join(config["data_path"], device["test_normal_data_path"])

        normal_data   = load_data(normal_path).sample(frac=1).reset_index(drop=True)
        abnormal_data = load_data(abnormal_path).sample(frac=1).reset_index(drop=True)
        new_normal    = load_data(test_np_path)

        n          = len(normal_data)
        train_size = int(0.4 * n)
        valid_size = int(0.1 * n)
        dev_size   = int(0.4 * n)
        test_start = train_size + valid_size + dev_size

        raw_clients.append({
            "device":       device["name"],
            "train_slice":  normal_data.iloc[:train_size],
            "valid_slice":  normal_data.iloc[train_size:train_size + valid_size],
            "test_slice":   normal_data.iloc[test_start:],
            "abnormal_data": abnormal_data,
            "new_normal":   new_normal,
        })

    logging.info("Loaded %d clients from %s (%s).", network_size, config_file, distribution)
    logging.info("Attack: data_poisoning | byz=%d%% | poison=%.0f%%",
                 byz_pct, POISONING_RATIO * 100)

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", torch_device)

    # ------------------------------------------------------------------ #
    # Outer loops — model types and independent runs                       #
    # ------------------------------------------------------------------ #
    for model_type in ["autoencoder", "hybrid"]:
        for run in range(num_runs):
            set_seeds(run * 10_000)

            # ---------------------------------------------------------- #
            # Assign Byzantine clients for this run                        #
            # ---------------------------------------------------------- #
            n_byzantine   = max(1, int(BYZANTINE_RATIO * network_size))
            byzantine_idx = set(random.sample(range(network_size), n_byzantine))
            byz_names     = [raw_clients[i]["device"] for i in byzantine_idx]
            logging.info("model=%s  run=%d  byzantine=%s", model_type, run, byz_names)

            # ---------------------------------------------------------- #
            # Build a single global scaler fitted on all CLEAN train data  #
            # (Byzantine clients' clean slices define the normal-traffic   #
            # distribution; poisoning only affects what's mixed in after)  #
            # ---------------------------------------------------------- #
            all_train_normal = pd.concat(
                [c["train_slice"] for c in raw_clients], ignore_index=True)
            all_valid_normal = pd.concat(
                [c["valid_slice"] for c in raw_clients], ignore_index=True)

            global_scaler           = IoTDataProccessor(scaler="standard")
            proc_train_clean, lbl_c = global_scaler.fit_transform(all_train_normal)
            proc_valid, lbl_v       = global_scaler.transform(all_valid_normal)

            # ---------------------------------------------------------- #
            # Replace Byzantine clients' training slices with poisoned     #
            # versions and rebuild the merged training array               #
            # ---------------------------------------------------------- #
            train_arrays = []
            for i, client in enumerate(raw_clients):
                proc_abn, _ = global_scaler.transform(
                    client["abnormal_data"], type="abnormal")

                if i in byzantine_idx:
                    arr = poison_train_slice(client["train_slice"], proc_abn, global_scaler)
                else:
                    arr, _ = global_scaler.transform(client["train_slice"])
                train_arrays.append(arr)

            proc_train_poisoned = np.concatenate(train_arrays, axis=0)
            lbl_train_poisoned  = np.zeros(len(proc_train_poisoned))

            train_loader = DataLoader(
                IoTDataset(proc_train_poisoned, lbl_train_poisoned),
                batch_size=batch_size, shuffle=True, pin_memory=True)
            valid_loader = DataLoader(
                IoTDataset(proc_valid, lbl_v),
                batch_size=batch_size, pin_memory=True)

            # ---------------------------------------------------------- #
            # Per-client test loaders (same structure as FL)               #
            # ---------------------------------------------------------- #
            client_test_loaders = []
            for client in raw_clients:
                proc_tn, lbl_tn = global_scaler.transform(client["test_slice"])
                proc_nn, lbl_nn = global_scaler.transform(client["new_normal"])
                proc_ab, lbl_ab = global_scaler.transform(
                    client["abnormal_data"], type="abnormal")

                proc_full = np.concatenate([proc_tn, proc_nn, proc_ab])
                lbl_full  = np.concatenate([lbl_tn,  lbl_nn,  lbl_ab])
                client_test_loaders.append(
                    DataLoader(IoTDataset(proc_full, lbl_full),
                               batch_size=batch_size, pin_memory=True))

            # ---------------------------------------------------------- #
            # Output file                                                   #
            # ---------------------------------------------------------- #
            result_dir = (
                f"Checkpoint/Results/{scen_name}/{network_size}"
                f"/{no_Exp}/Run_{run}/{metric}"
            )
            os.makedirs(result_dir, exist_ok=True)
            result_file = f"{result_dir}/{scen_name}_{model_type}_results.json"
            open(result_file, "w").close()

            # ---------------------------------------------------------- #
            # Build model                                                   #
            # ---------------------------------------------------------- #
            if model_type == "hybrid":
                model = Shrink_Autoencoder(
                    input_dim=dim_features, output_dim=dim_features,
                    shrink_lambda=shrink_lambda, latent_dim=11, hidden_neus=50)
            else:
                model = Autoencoder(
                    input_dim=dim_features, output_dim=dim_features,
                    latent_dim=11, hidden_neus=50)

            model     = model.to(torch_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

            ckpt_dir  = (f"Checkpoint/{network_size}/{no_Exp}/Run_{run}"
                         f"/{scen_name}/{model_type}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "best_model.cpt")

            min_val_loss  = float("inf")
            worse_count   = 0
            early_stopped = False
            client_latent = {}

            # ---------------------------------------------------------- #
            # Training loop                                                 #
            # ---------------------------------------------------------- #
            for phase in range(num_phases):
                if early_stopped:
                    break

                phase_train_loss = 0.0
                phase_val_loss   = 0.0

                for ep in range(epoch):
                    t_loss = train_one_epoch(model, train_loader, optimizer, torch_device)
                    v_loss = validate_epoch(model, valid_loader, torch_device)

                    logging.info(
                        "[%s | run=%d | phase %d/%d | ep %d/%d]  "
                        "train=%.6f  val=%.6f",
                        model_type, run, phase + 1, num_phases,
                        ep + 1, epoch, t_loss, v_loss)

                    phase_train_loss = t_loss
                    phase_val_loss   = v_loss

                    if v_loss < min_val_loss:
                        min_val_loss = v_loss
                        worse_count  = 0
                        torch.save(model.state_dict(), ckpt_path)
                    else:
                        worse_count += 1
                        if worse_count >= patience:
                            logging.info("Early stopping at phase %d ep %d.",
                                         phase + 1, ep + 1)
                            early_stopped = True
                            break

                # Phase-end evaluation on all clients
                evaluator = Evaluator(model, metric=metric, model_type=model_type)
                phase_results  = {}
                client_latent[phase] = {}

                for client, test_loader in zip(raw_clients, client_test_loaders):
                    if model_type == "hybrid":
                        auc, latent, lbl = evaluator.evaluate(
                            test_loader, train_loader)
                        client_latent[phase][client["device"]] = (latent, lbl)
                    else:
                        auc = evaluator.evaluate(test_loader, train_loader)
                    phase_results[client["device"]] = auc

                phase_results["val_loss"]        = phase_val_loss
                phase_results["train_loss"]      = phase_train_loss
                phase_results["byzantine_ratio"] = BYZANTINE_RATIO
                phase_results["poisoning_ratio"] = POISONING_RATIO
                phase_results["byzantine_clients"] = byz_names
                phase_results["attack_type"]     = "data_poisoning"

                with open(result_file, "a") as f:
                    f.write(json.dumps({f"phase_{phase + 1}": phase_results}) + "\n")

                logging.info("Phase %d/%d done. val_loss=%.6f",
                             phase + 1, num_phases, phase_val_loss)

            if model_type == "hybrid":
                latent_path = (
                    f"Checkpoint/LatentData/{scen_name}/{network_size}"
                    f"/{no_Exp}/Run_{run}/latent_{model_type}.pkl"
                )
                os.makedirs(os.path.dirname(latent_path), exist_ok=True)
                with open(latent_path, "wb") as f:
                    pickle.dump(client_latent, f)

            logging.info("Saved → %s", result_file)
