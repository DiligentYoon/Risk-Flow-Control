from __future__ import annotations

import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import yaml

from isaaclab.app import AppLauncher

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Train one-step TD safety value function.")
parser.add_argument("--dataset", type=str, required=True, help="Path to processed HDF5 dataset.")
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from tasks.p02_safety_value.agent.safety import Safety
from tasks.p02_safety_value.buffer.dataset import SafetyValueDataset
from tasks.p02_safety_value.model.safety import SafetyCritic

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    cfg_path = os.path.abspath("src/tasks/p02_safety_value/envs/R1/R1_fall/cfg/predictor_cfg.yaml") # fixed
    with open(cfg_path, "r", encoding="utf-8") as f:
        pred_cfg = yaml.safe_load(f)

    seed = args_cli.seed if args_cli.seed is not None else pred_cfg["seed"]
    set_seed(seed)
    pred_cfg["agent"]["seed"] = seed

    device = torch.device("cuda")
    dataset_path = os.path.abspath(args_cli.dataset)
    log_dir = os.path.join(os.path.dirname(dataset_path), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    checkpoint_dir = os.path.join(log_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(log_dir, "predictor_cfg.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(pred_cfg, f, sort_keys=False)

    # ====================== Dataset ====================== #
    train_dataset = SafetyValueDataset(dataset_path=dataset_path, split="train")
    validation_dataset = SafetyValueDataset(dataset_path=dataset_path, split="validation")
    test_dataset = SafetyValueDataset(dataset_path=dataset_path, split="test")

    batch_size = pred_cfg["agent"]["batch_size"]
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

    num_states = train_dataset.states.shape[-1]

    # ====================== Model & Agent ====================== #
    pred_model = {"critic_1": SafetyCritic(num_states=num_states, device=device),
                  "critic_2": SafetyCritic(num_states=num_states, device=device)}
    pred_agent = Safety(model=pred_model, device=device, cfg=pred_cfg["agent"])

    # ====================== Training Setup ====================== #
    total_updates = pred_cfg["train"]["timesteps"]
    eval_threshold = pred_cfg["eval"]["threshold"]
    CLI_interval = 100
    eval_interval = 1000
    checkpoint_interval = int(total_updates / 3)

    writer = SummaryWriter(log_dir=log_dir)

    print(f"[INFO] Dataset: {dataset_path}")
    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Validation samples: {len(validation_dataset)}")
    print(f"[INFO] Test samples: {len(test_dataset)}")
    print(f"[INFO] Log directory: {log_dir}")

    # ====================== Training ====================== #
    train_iterator = iter(train_loader)
    start_time = time.time()
    best_accuracy = -float("inf")

    for update in range(1, total_updates + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        train_metrics = pred_agent.update(batch)

        if update % CLI_interval == 0:
            elapsed_time = time.time() - start_time
            for key, value in train_metrics.items():
                writer.add_scalar(f"Train/{key}", value, update)

            print(
                f"[TRAIN] update={update}/{total_updates} | "
                f"loss={train_metrics['loss']:.5f} | "
                f"loss_1={train_metrics['loss_1']:.5f} | "
                f"loss_2={train_metrics['loss_2']:.5f} | "
                f"value_mean={train_metrics['value_mean']:.4f} | "
                f"target_mean={train_metrics['target_mean']:.4f} | "
                f"gap={train_metrics['critic_gap']:.4f}"
            )

        if update % eval_interval == 0:
            eval_metrics = pred_agent.evaluate(validation_loader, threshold=eval_threshold)

            for key, value in eval_metrics.items():
                writer.add_scalar(f"Validation/{key}", value, update)

            print(
                f"[EVAL] update={update}/{total_updates} | "
                f"td_loss={eval_metrics['td_loss']:.5f} | "
                f"gap={eval_metrics['critic_gap']:.4f} | "
                f"RCR={eval_metrics['risk_coverage_rate']:.4f} | "
                f"RDR={eval_metrics['risk_detection_rate']:.4f} | "
                f"RFAR={eval_metrics['risk_false_alarm_rate']:.4f} | "
                f"Termination_DR={eval_metrics['termination_detection_rate']:.4f} | "
                f"Termination_FAR={eval_metrics['termination_false_alarm_rate']:.4f} | "
                f"PR={eval_metrics['proactive_recall']:.4f} | "
                f"Accuracy={eval_metrics['accuracy']:.4f} | "
                f"pred_risk={eval_metrics['pred_risk_rate']:.4f} | "
                f"real_risk={eval_metrics['real_risk_rate']:.4f}"
            )

            if eval_metrics["accuracy"] > best_accuracy:
                best_accuracy = eval_metrics["accuracy"]
                pred_agent.save(os.path.join(checkpoint_dir, "agent_best.pt"))

        if update % checkpoint_interval == 0:
            pred_agent.save(os.path.join(checkpoint_dir, f"agent_{update}.pt"))

    # ====================== Final Evaluation ====================== #
    final_checkpoint_path = os.path.join(checkpoint_dir, "agent_final.pt")
    pred_agent.save(final_checkpoint_path)

    test_metrics = pred_agent.evaluate(test_loader, threshold=eval_threshold)
    for key, value in test_metrics.items():
        writer.add_scalar(f"Test/{key}", value, total_updates)

    writer.close()

    print(
        f"[TEST] "
        f"td_loss={test_metrics['td_loss']:.5f} | "
        f"Risk coverage rate={test_metrics['risk_coverage_rate'] * 100:.2f}% | "
        f"Risk Detection rate={test_metrics['risk_detection_rate'] * 100:.2f}% | "
        f"Risk False alarm rate={test_metrics['risk_false_alarm_rate'] * 100:.2f}% | "
        f"Termination detection rate={test_metrics['termination_detection_rate'] * 100:.2f}% | "
        f"Termination false alarm rate={test_metrics['termination_false_alarm_rate'] * 100:.2f}% | "
        f"Proactive recall={test_metrics['proactive_recall']* 100:.2f} | "
        f"Accuracy={test_metrics['accuracy'] * 100:.2f}% | "
        f"pred_risk={test_metrics['pred_risk_rate']:.4f} | "
        f"real_risk={test_metrics['real_risk_rate']:.4f}"
    )
    print(f"[INFO] Training completed.")
    print(f"[INFO] Final checkpoint: {final_checkpoint_path}")


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()