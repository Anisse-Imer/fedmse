"""
Centralized training baseline for comparison with federated learning.

Strategy:
  1. Load all client data from the same config as main.py.
  2. Apply the same per-client normal-data split as FL (40% train / 10% val /
     skip dev / 10% test) so the raw samples used for training and testing are
     identical between the two approaches.
  3. Merge the train and validation slices across all clients into two global
     datasets (train, valid).  The per-client test slices + abnormal + new-normal
     data are kept separate so AUC can be reported per-device — exactly matching
     the FL output structure.
  4. Fit ONE global StandardScaler on the merged training data (vs. a per-client
     scaler in FL).
  5. Train a single global model directly (no federation, no aggregation).
  6. Evaluate every `epoch` training epochs ("phase") and write JSON results in
     the same newline-delimited format as main.py — with "phase_N" keys so
     results from both scripts can be loaded and compared side-by-side.
"""

import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import DataLoader

from Model import Shrink_Autoencoder, Autoencoder
from DataLoader import load_data, IoTDataset, IoTDataProccessor
from Evaluator import Evaluator

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# CLI — accept a config file path so runs can be scripted without editing source
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--file", "-f",
    required=True,
    help="Path to the JSON configuration file (relative to src/).",
)
_args = _parser.parse_args()

# ---------------------------------------------------------------------------
# Hyper-parameters — kept identical to main.py for a fair comparison
# ---------------------------------------------------------------------------
epoch          = 100          # training epochs per evaluation phase
num_phases     = 20           # evaluation checkpoints (≈ FL rounds)
lr_rate        = 1e-5
shrink_lambda  = 10
data_seed      = 1234
batch_size     = 12
dim_features   = 115          # N-BaIoT: 115 features
metric         = "AUC"
patience       = 3            # early-stopping patience (epochs)
scen_name      = "Centralized"

config_file = _args.file

# network_size is derived from the config so --file is the only argument needed
with open(config_file) as _f:
    _cfg_preview = json.load(_f)
network_size = len(_cfg_preview["devices_list"])

# Derive a short distribution label from the config filename so that IID and
# non-IID runs always write to separate directories and never overwrite each other.
_cfg_stem = os.path.splitext(os.path.basename(config_file))[0]
distribution = "nonIID" if "non-iid" in _cfg_stem.lower() else "IID"

no_Exp = (
    f"Centralized_{distribution}_{epoch}epoch_{network_size}client_"
    f"{num_phases}phases_lr{lr_rate}_lamda{shrink_lambda}_dataseed{data_seed}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_input in train_loader:
        _, _, loss = model(batch_input[0].to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate_epoch(model, valid_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_input in valid_loader:
            _, _, loss = model(batch_input[0].to(device))
            total_loss += loss.item()
    return total_loss / len(valid_loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seeds(data_seed)

    # ------------------------------------------------------------------ #
    # 1. Load configuration                                                #
    # ------------------------------------------------------------------ #
    logging.info("Loading configuration from %s …", config_file)
    with open(config_file, "r") as f:
        config = json.load(f)

    devices_list = config["devices_list"][:network_size]

    # ------------------------------------------------------------------ #
    # 2. Load and split data (same ratios as FL main.py)                  #
    # ------------------------------------------------------------------ #
    all_train_normal  = []   # 40 % of each client's normal data (merged → Dataset 1: Train)
    all_valid_normal  = []   # 10 % of each client's normal data (merged → Dataset 2: Validation)
    client_info       = []   # per-client test data kept separate for evaluation

    for device in devices_list:
        normal_path      = os.path.join(config["data_path"], device["normal_data_path"])
        abnormal_path    = os.path.join(config["data_path"], device["abnormal_data_path"])
        test_normal_path = os.path.join(config["data_path"], device["test_normal_data_path"])

        # Shuffle exactly like FL (no fixed random_state → uses global seed)
        normal_data   = load_data(normal_path  ).sample(frac=1).reset_index(drop=True)
        abnormal_data = load_data(abnormal_path).sample(frac=1).reset_index(drop=True)
        new_normal_data = load_data(test_normal_path)

        n = len(normal_data)
        train_size = int(0.4 * n)
        valid_size = int(0.1 * n)
        # dev (40 %) is skipped — not needed without aggregation
        # test: last 10 % (= n - train - valid - dev)
        dev_size   = int(0.4 * n)
        test_start = train_size + valid_size + dev_size

        train_slice = normal_data.iloc[:train_size]
        valid_slice = normal_data.iloc[train_size : train_size + valid_size]
        test_slice  = normal_data.iloc[test_start:]          # Dataset 3: Test (normal part)

        all_train_normal.append(train_slice)
        all_valid_normal.append(valid_slice)

        logging.info(
            "%s — train: %d  val: %d  test-normal: %d  abnormal: %d  new-normal: %d",
            device["name"], len(train_slice), len(valid_slice),
            len(test_slice), len(abnormal_data), len(new_normal_data),
        )

        client_info.append({
            "device":          device["name"],
            "test_slice":      test_slice,
            "abnormal_data":   abnormal_data,
            "new_normal_data": new_normal_data,
        })

    # ------------------------------------------------------------------ #
    # 3. Merge train / val across clients and fit a single global scaler  #
    # ------------------------------------------------------------------ #
    merged_train_normal = pd.concat(all_train_normal, ignore_index=True)
    merged_valid_normal = pd.concat(all_valid_normal, ignore_index=True)

    logging.info(
        "Merged — train: %d samples | valid: %d samples",
        len(merged_train_normal), len(merged_valid_normal),
    )

    global_scaler = IoTDataProccessor(scaler="standard")
    proc_train, train_label = global_scaler.fit_transform(merged_train_normal)
    proc_valid, valid_label = global_scaler.transform(merged_valid_normal)

    train_dataset = IoTDataset(proc_train, train_label)
    valid_dataset = IoTDataset(proc_valid, valid_label)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True)

    # ------------------------------------------------------------------ #
    # 4. Build per-client test loaders using the global scaler            #
    #    Structure mirrors FL: test_normal + new_normal + abnormal        #
    # ------------------------------------------------------------------ #
    for client in client_info:
        proc_test_normal, lbl_test  = global_scaler.transform(client["test_slice"])
        proc_new_normal,  lbl_new   = global_scaler.transform(client["new_normal_data"])
        proc_abnormal,    lbl_abn   = global_scaler.transform(client["abnormal_data"], type="abnormal")

        proc_test  = np.concatenate([proc_test_normal, proc_new_normal, proc_abnormal], axis=0)
        lbl_test_f = np.concatenate([lbl_test, lbl_new, lbl_abn], axis=0)

        client["test_loader"] = DataLoader(
            IoTDataset(proc_test, lbl_test_f), batch_size=batch_size, pin_memory=True
        )

    # ------------------------------------------------------------------ #
    # 5. Training loop                                                     #
    # ------------------------------------------------------------------ #
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", torch_device)

    for model_type in ["autoencoder", "hybrid"]:
        set_seeds(data_seed)
        logging.info("=== model_type=%s ===", model_type)

        # Output paths (parallel structure to FL's Checkpoint/Results/Update/…)
        result_dir = (
            f"Checkpoint/Results/{scen_name}/{network_size}/{no_Exp}/{metric}"
        )
        os.makedirs(result_dir, exist_ok=True)
        result_file = f"{result_dir}/{scen_name}_{model_type}_results.json"
        open(result_file, "w").close()   # reset file

        # Model
        if model_type == "hybrid":
            model = Shrink_Autoencoder(
                input_dim=dim_features, output_dim=dim_features,
                shrink_lambda=shrink_lambda, latent_dim=11, hidden_neus=50,
            )
        else:
            model = Autoencoder(
                input_dim=dim_features, output_dim=dim_features,
                latent_dim=11, hidden_neus=50,
            )

        model = model.to(torch_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

        min_val_loss   = float("inf")
        worse_count    = 0
        early_stopped  = False
        client_latent  = {}   # only populated for hybrid

        # Best-model checkpoint path
        ckpt_dir = f"Checkpoint/{network_size}/{no_Exp}/{scen_name}/{model_type}"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "best_model.cpt")

        for phase in range(num_phases):
            if early_stopped:
                break

            phase_train_loss = 0.0
            phase_val_loss   = 0.0

            # -- inner epoch loop (one "phase" = `epoch` training epochs) --
            for ep in range(epoch):
                t_loss = train_one_epoch(model, train_loader, optimizer, torch_device)
                v_loss = validate_epoch(model, valid_loader, torch_device)

                logging.info(
                    "[%s | phase %d/%d | ep %d/%d]  train=%.6f  val=%.6f",
                    model_type, phase + 1, num_phases, ep + 1, epoch, t_loss, v_loss,
                )

                phase_train_loss = t_loss
                phase_val_loss   = v_loss

                if v_loss < min_val_loss:
                    min_val_loss = v_loss
                    worse_count  = 0
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    worse_count += 1
                    if worse_count >= patience:
                        logging.info("Early stopping at phase %d, epoch %d.", phase + 1, ep + 1)
                        early_stopped = True
                        break

            # -- phase-end evaluation on all clients --
            evaluator      = Evaluator(model, metric=metric, model_type=model_type)
            phase_results  = {}
            client_latent[phase] = {}

            for client in client_info:
                if model_type == "hybrid":
                    # train_loader used to fit the centroid on the global training latent
                    auc_score, test_latent, test_lbl = evaluator.evaluate(
                        client["test_loader"], train_loader
                    )
                    client_latent[phase][client["device"]] = (test_latent, test_lbl)
                else:
                    auc_score = evaluator.evaluate(client["test_loader"], train_loader)

                phase_results[client["device"]] = auc_score
                logging.info("  %s → AUC = %.4f", client["device"], auc_score)

            phase_results["val_loss"]   = phase_val_loss
            phase_results["train_loss"] = phase_train_loss

            with open(result_file, "a") as f:
                f.write(json.dumps({f"phase_{phase + 1}": phase_results}) + "\n")

            logging.info(
                "Phase %d/%d done.  val_loss=%.6f", phase + 1, num_phases, phase_val_loss
            )

        # -- save latent data for hybrid (same structure as FL) --
        if model_type == "hybrid":
            latent_path = (
                f"Checkpoint/LatentData/{scen_name}/{network_size}/"
                f"{no_Exp}/latent_{model_type}.pkl"
            )
            os.makedirs(os.path.dirname(latent_path), exist_ok=True)
            with open(latent_path, "wb") as f:
                pickle.dump(client_latent, f)
            logging.info("Latent data saved to %s", latent_path)

        logging.info("Results saved to %s", result_file)
