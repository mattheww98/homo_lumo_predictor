from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader


from model import GapPredictor
from utils import (
        Standardizer,
        count_parameters,
        fit_node_standardizer,
        fit_target_standardizer,
        load_config,
        resolve_path,
        seed_all,
        split_dataset
    )

def make_model(config: dict[str, Any], node_scaler: Standardizer):
    data_cfg = config["data"]
    model_cfg = config["model"]
    rbf_cfg = model_cfg["rbf"]
    return GapPredictor(
        node_feat_indices=data_cfg["node_feature_indices"],
        edge_feat_indices=data_cfg["bond_feature_indices"],
        node_dim=model_cfg["node_dim"],
        edge_dim=model_cfg["edge_dim"],
        global_dim=model_cfg["global_dim"],
        block_hidden_dim=model_cfg["block_hidden_dim"],
        num_blocks=model_cfg["num_blocks"],
        rbf_min=rbf_cfg["min_centre"],
        rbf_max=rbf_cfg["max_centre"],
        num_rbf=rbf_cfg["num_centres"],
        sigma_scale=rbf_cfg["sigma_multiplier"],
        head_hidden_dim=model_cfg["prediction_head_hidden_dim"],
        node_mean=node_scaler.mean,
        node_std=node_scaler.std
    )


def _target(batch: Any, target_index: int):
    return batch.y.reshape(batch.num_graphs, -1)[:, target_index]


def train_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    target_scaler: Standardizer,
    target_index: int,
    device: torch.device
):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    criterion = nn.MSELoss()

    for batch in loader:
        batch = batch.to(device)
        targets = target_scaler.scale(_target(batch, target_index))
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Any,
    target_scaler: Standardizer,
    target_index: int,
    device: torch.device,
    collect_predictions: bool = False
):
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    count = 0
    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        targets = _target(batch, target_index)
        predictions = target_scaler.unscale(model(batch))
        squared_error += torch.sum((predictions - targets) ** 2).item()
        absolute_error += torch.sum(torch.abs(predictions - targets)).item()
        count += targets.numel()
        if collect_predictions:
            all_targets.append(targets.detach().cpu())
            all_predictions.append(predictions.detach().cpu())
    if count == 0:
        metrics: dict[str, Any] = {"mse": float("nan"), "mae": float("nan")}
    else:
        metrics = {"mse": squared_error / count, "mae": absolute_error / count}
    if collect_predictions:
        metrics["targets"] = torch.cat(all_targets) if all_targets else torch.empty(0)
        metrics["predictions"] = (
            torch.cat(all_predictions) if all_predictions else torch.empty(0)
        )
    return metrics


def run_training(config_path: str | Path = Path(__file__).with_name("config.json")):

    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    seed = int(config["seed"])
    seed_all(seed)

    data_cfg = config["data"]
    training_cfg = config["training"]
    dataset_root = resolve_path(data_cfg["root"], config_path.parent)
    dataset = QM9(root=str(dataset_root))
    train_set, val_set, test_set = split_dataset(
        dataset,
        train_size=int(data_cfg["train_size"]),
        val_size=int(data_cfg["val_size"]),
        test_size=int(data_cfg["test_size"]),
        seed=seed
    )
    if len(train_set) == 0:
        raise ValueError("Training dataset cannot be empty")

    node_scaler = fit_node_standardizer(
        train_set,
        data_cfg["node_feature_indices"],
        data_cfg.get("continuous_node_feature_indices", []),
    )
    target_index = int(data_cfg["target_index"])
    target_scaler = fit_target_standardizer(train_set, target_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": int(training_cfg["batch_size"]),
        "num_workers": int(training_cfg.get("num_workers", 0)),
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    model = make_model(config, node_scaler).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    epochs = int(training_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(training_cfg["learning_rate"]),
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=float(training_cfg.get("one_cycle_pct_start", 0.3)),
        anneal_strategy=training_cfg.get("one_cycle_anneal_strategy", "cos"),
    )

    print(f"Device: {device}; trainable parameters: {count_parameters(model):,}")
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    best_scheduler_state: dict[str, Any] | None = None
    stale_epochs = 0
    patience = int(training_cfg["early_stopping_patience"])
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_mse = train_epoch(
            model, train_loader, optimizer, scheduler, target_scaler, target_index, device
        )
        train_metrics = evaluate(model, train_loader, target_scaler, target_index, device)
        val_metrics = evaluate(model, val_loader, target_scaler, target_index, device)
        history.append(
            {
                "epoch": epoch,
                "train_scaled_mse": train_mse,
                "train_mae": train_metrics["mae"],
                "train_mse": train_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "val_mse": val_metrics["mse"],
            }
        )
        print(
            f"Epoch {epoch:03d}/{epochs}: "
            f"train MAE={train_metrics['mae']:.6f}, train MSE={train_metrics['mse']:.6f}; "
            f"val MAE={val_metrics['mae']:.6f}, val MSE={val_metrics['mse']:.6f}; "
            f"lr={scheduler.get_last_lr()[0]:.3e}"
        )

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_scheduler_state = copy.deepcopy(scheduler.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

    if best_state is None: 
        best_state = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        best_scheduler_state = copy.deepcopy(scheduler.state_dict())
        best_epoch = len(history)
    model.load_state_dict(best_state)
    test_results = evaluate(
        model,
        test_loader,
        target_scaler,
        target_index,
        device,
        collect_predictions=True,
    )
    test_targets = test_results.pop("targets")
    test_predictions = test_results.pop("predictions")
    test_metrics = test_results

    checkpoint_path = resolve_path(config["output"]["checkpoint_path"], config_path.parent)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": best_optimizer_state,
            "scheduler_state_dict": best_scheduler_state,
            "config": config,
            "node_standardizer": node_scaler.state_dict(),
            "target_standardizer": target_scaler.state_dict(),
            "best_epoch": best_epoch,
            "best_val_mae": best_mae,
            "test_metrics": test_metrics,
            "test_targets": test_targets,
            "test_predictions": test_predictions,
            "history": history,
        },
        checkpoint_path,
    )
    print(f"Best epoch: {best_epoch}; test MAE={test_metrics['mae']:.6f}")
    print(f"Saved checkpoint to {checkpoint_path}")
    return {"best_epoch": best_epoch, "best_val_mae": best_mae, "test": test_metrics, "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
        help="Path to the JSON configuration file"
    )
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
