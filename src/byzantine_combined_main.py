"""
Federated Learning with combined Byzantine attack simulation.

Attack types (applied simultaneously on Byzantine clients):
  - Data poisoning   : 30 % of training samples replaced with abnormal traffic.
  - Model corruption : after local training the weight update (delta) is
                       sign-flipped and scaled, sending a destructive gradient
                       to FedAvg instead of the honest update.

Fixed Byzantine ratio: 30 % of clients.
Network size        : 30 clients (IID).

Usage:
  python byzantine_combined_main.py \
      --file Configuration/scen2-nba-iot-byzantine-30clients-iid.json
"""

import os
import json
import pickle
import argparse
import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from Model import Shrink_Autoencoder, Autoencoder
from DataLoader import load_data, IoTDataset, IoTDataProccessor
from Trainer import ClientTrainer, GlobalAggregator
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
# Fixed hyper-parameters
# ---------------------------------------------------------------------------
epoch            = 100
num_rounds       = 20
num_runs         = 5
lr_rate          = 1e-5
shrink_lambda    = 10
data_seed        = 1234
batch_size       = 12
dim_features     = 115
metric           = "AUC"
update_type      = "mse_avg"
num_participants = 0.5
scen_name        = "Byzantine"

# Attack settings — fixed for this experiment
BYZANTINE_RATIO  = 0.3   # 30 % of clients are Byzantine
POISONING_RATIO  = 0.3   # 30 % of a Byzantine client's training set is abnormal
# Scaling factor applied to the sign-flipped weight update for model corruption.
# A factor of 1.0 is a pure sign-flip; larger values amplify the disruption.
CORRUPTION_SCALE = 1.0

config_file  = args.file
_cfg_stem    = os.path.splitext(os.path.basename(config_file))[0]
distribution = "nonIID" if "non-iid" in _cfg_stem.lower() else "IID"

with open(config_file) as f:
    _cfg = json.load(f)
network_size = len(_cfg["devices_list"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_poisoned_loader(proc_train, proc_abnormal, batch_size):
    """Replace POISONING_RATIO fraction of training samples with abnormal ones."""
    n_total  = len(proc_train)
    n_poison = max(1, int(n_total * POISONING_RATIO))
    n_clean  = n_total - n_poison

    idx    = np.random.choice(len(proc_abnormal), n_poison,
                              replace=(n_poison > len(proc_abnormal)))
    mixed  = np.concatenate([proc_train[:n_clean], proc_abnormal[idx]], axis=0)
    labels = np.zeros(len(mixed))
    return DataLoader(IoTDataset(mixed, labels),
                      batch_size=batch_size, shuffle=True, pin_memory=True)


def corrupt_weights(local_state: dict, global_state: dict) -> dict:
    """
    Model corruption: sign-flip the weight update (delta) and scale it.

    Sends  global_weights - CORRUPTION_SCALE * (local_weights - global_weights)
    instead of local_weights, pushing the global model in the opposite direction.
    """
    corrupted = {}
    for key in local_state:
        delta = local_state[key].float() - global_state[key].float()
        corrupted[key] = (global_state[key].float()
                          - CORRUPTION_SCALE * delta).to(local_state[key].dtype)
    return corrupted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_seeds(data_seed)

    byz_pct = int(BYZANTINE_RATIO * 100)
    no_Exp  = (
        f"Byzantine_{distribution}_combined_byz{byz_pct}pct_"
        f"{epoch}epoch_{network_size}client_{num_rounds}rounds_"
        f"lr{lr_rate}_lamda{shrink_lambda}_dataseed{data_seed}"
    )

    # ------------------------------------------------------------------ #
    # Load data once                                                        #
    # ------------------------------------------------------------------ #
    with open(config_file) as f:
        config = json.load(f)

    client_info = []
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

        train_normal = normal_data.iloc[:train_size]
        valid_normal = normal_data.iloc[train_size:train_size + valid_size]
        dev_normal   = normal_data.iloc[train_size + valid_size:train_size + valid_size + dev_size]
        test_normal  = normal_data.iloc[train_size + valid_size + dev_size:]

        proc = IoTDataProccessor(scaler="standard")
        proc_train,    lbl_train    = proc.fit_transform(train_normal)
        proc_valid,    lbl_valid    = proc.transform(valid_normal)
        proc_test,     lbl_test     = proc.transform(test_normal)
        proc_new_norm, lbl_new_norm = proc.transform(new_normal)
        proc_abnormal, lbl_abnormal = proc.transform(abnormal_data, type="abnormal")

        proc_test_full = np.concatenate([proc_test, proc_new_norm, proc_abnormal])
        lbl_test_full  = np.concatenate([lbl_test,  lbl_new_norm,  lbl_abnormal])

        client_info.append({
            "device":        device["name"],
            "train_loader":  DataLoader(IoTDataset(proc_train, lbl_train),
                                        batch_size=batch_size, pin_memory=True),
            "valid_loader":  DataLoader(IoTDataset(proc_valid, lbl_valid),
                                        batch_size=batch_size, pin_memory=True),
            "test_loader":   DataLoader(IoTDataset(proc_test_full, lbl_test_full),
                                        batch_size=batch_size, pin_memory=True),
            "proc_train":    proc_train,
            "proc_abnormal": proc_abnormal,
            "dev_normal":    dev_normal,
            "is_byzantine":  False,
            "save_dir":      "",
        })

    logging.info("Loaded %d clients from %s (%s).", len(client_info), config_file, distribution)
    logging.info("Attack: data_poisoning + model_corruption | byz=%d%% | poison=%.0f%%",
                 byz_pct, POISONING_RATIO * 100)

    # ------------------------------------------------------------------ #
    # Experiment loop — model types and independent runs                    #
    # ------------------------------------------------------------------ #
    for model_type in ["autoencoder", "hybrid"]:
        for run in range(num_runs):
            set_seeds(run * 10_000)

            # Assign Byzantine clients for this run
            n_byzantine   = max(1, int(BYZANTINE_RATIO * len(client_info)))
            byzantine_idx = set(random.sample(range(len(client_info)), n_byzantine))
            for i, client in enumerate(client_info):
                client["is_byzantine"] = (i in byzantine_idx)
                client["save_dir"] = os.path.join(
                    f"Checkpoint/{network_size}/{no_Exp}/Run_{run}/ClientModel",
                    scen_name, model_type, update_type, client["device"]
                )

            byz_names = [client_info[i]["device"] for i in byzantine_idx]
            logging.info("model=%s  run=%d  byzantine=%s", model_type, run, byz_names)

            # Output file
            result_dir = (
                f"Checkpoint/Results/{scen_name}/{network_size}"
                f"/{no_Exp}/Run_{run}/{metric}"
            )
            os.makedirs(result_dir, exist_ok=True)
            result_file = (
                f"{result_dir}/{scen_name}_{num_participants}"
                f"_{model_type}_{update_type}_results.json"
            )
            open(result_file, "w").close()

            # Build global model
            if model_type == "hybrid":
                global_model = Shrink_Autoencoder(
                    input_dim=dim_features, output_dim=dim_features,
                    shrink_lambda=shrink_lambda, latent_dim=11, hidden_neus=50)
            else:
                global_model = Autoencoder(
                    input_dim=dim_features, output_dim=dim_features,
                    latent_dim=11, hidden_neus=50)

            global_aggregator = GlobalAggregator(global_model, update_type=update_type)

            min_len  = min(len(c["dev_normal"]) for c in client_info)
            dev_data = np.concatenate(
                [c["dev_normal"].sample(n=min_len) for c in client_info], axis=0)
            global_aggregator.create_dev_dataset({"dataset": dev_data})

            min_val_loss  = float("inf")
            global_worse  = 0
            client_latent = {}

            for rnd in range(num_rounds):
                client_latent[rnd] = {}

                # Snapshot global weights before any local training this round
                global_state = copy.deepcopy(global_aggregator.model.state_dict())

                selected_idx     = random.sample(range(len(client_info)),
                                                 int(num_participants * len(client_info)))
                selected_clients = [client_info[i] for i in selected_idx]
                total_samples    = sum(len(c["train_loader"].dataset)
                                       for c in selected_clients)

                client_weights = []
                rnd_byzantine  = []

                for client in selected_clients:
                    is_byz = client["is_byzantine"]

                    # Data poisoning: replace part of training set with abnormal samples
                    train_loader = (
                        build_poisoned_loader(
                            client["proc_train"], client["proc_abnormal"], batch_size)
                        if is_byz else client["train_loader"]
                    )

                    trainer = ClientTrainer(
                        model=global_aggregator.model,
                        save_dir=client["save_dir"],
                        epoch=epoch,
                        lr_rate=lr_rate,
                        update_type=update_type,
                    )
                    trainer.run(train_loader, client["valid_loader"])

                    local_state = copy.deepcopy(trainer.model.state_dict())

                    # Model corruption: sign-flip the weight update for Byzantine clients
                    if is_byz:
                        local_state = corrupt_weights(local_state, global_state)
                        rnd_byzantine.append(client["device"])

                    client_weights.append((
                        local_state,
                        total_samples,
                        len(client["train_loader"].dataset),
                    ))

                global_aggregator.update(local_models=client_weights)

                evaluator     = Evaluator(global_aggregator.model,
                                         metric=metric, model_type=model_type)
                round_results = {}
                for client in client_info:
                    if model_type == "hybrid":
                        auc, latent, lbl = evaluator.evaluate(
                            client["test_loader"], client["train_loader"])
                        client_latent[rnd][client["device"]] = (latent, lbl)
                    else:
                        auc = evaluator.evaluate(
                            client["test_loader"], client["train_loader"])
                    round_results[client["device"]] = auc

                round_results["global_loss"]       = global_aggregator.val_loss
                round_results["join_clients"]       = selected_idx
                round_results["byzantine_clients"]  = rnd_byzantine
                round_results["byzantine_ratio"]    = BYZANTINE_RATIO
                round_results["poisoning_ratio"]    = POISONING_RATIO
                round_results["corruption_scale"]   = CORRUPTION_SCALE
                round_results["attack_type"]        = "data_poisoning+model_corruption"

                with open(result_file, "a") as f:
                    f.write(json.dumps({f"round_{rnd + 1}": round_results}) + "\n")

                logging.info("model=%s  run=%d  rnd=%d/%d  val_loss=%.6f",
                             model_type, run, rnd + 1, num_rounds,
                             global_aggregator.val_loss)

                if global_aggregator.val_loss < min_val_loss:
                    min_val_loss = global_aggregator.val_loss
                    global_worse = 0
                else:
                    global_worse += 1
                    if global_worse > 1:
                        logging.info("Early stopping at round %d.", rnd + 1)
                        break

            if model_type == "hybrid":
                latent_path = (
                    f"Checkpoint/LatentData/{scen_name}/{network_size}"
                    f"/{no_Exp}/Run_{run}/latent_{model_type}_{update_type}.pkl"
                )
                os.makedirs(os.path.dirname(latent_path), exist_ok=True)
                with open(latent_path, "wb") as f:
                    pickle.dump(client_latent, f)

            logging.info("Saved → %s", result_file)
