"""Fixed linear probe for first-appearance ordered room content."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tinymem.research.readout_controls import state_donors
from tinymem.research.update_protocol import file_sha256
from scripts.readout_content_validation import validate_exports

L2 = 0.001
MAX_ITER = 300


def fit_heads(features, targets):
    x = torch.as_tensor(features, dtype=torch.float64, device="cpu")
    y = torch.as_tensor(targets, dtype=torch.long, device="cpu")
    if x.ndim != 2 or x.shape[1] != 16 or y.shape != (len(x), 8):
        raise ValueError("expected N by 16 features and N by 8 targets")
    if not torch.isfinite(x).all() or ((y < 0) | (y >= 6)).any():
        raise ValueError("invalid features or room targets")
    mean, scale = x.mean(0), x.std(0, correction=0)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    z = (x - mean) / scale
    weights = torch.zeros(16, 8, 6, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(8, 6, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights, bias], lr=1, max_iter=MAX_ITER,
        tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = torch.einsum("nd,dkc->nkc", z, weights) + bias
        loss = F.cross_entropy(logits.reshape(-1, 6), y.reshape(-1)) + L2 * weights.square().sum() / 16
        loss.backward()
        return loss

    optimizer.step(closure)
    objective = float(closure().detach())
    gradient = max(float(weights.grad.abs().max()), float(bias.grad.abs().max()))
    return {"mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.detach().tolist(),
            "bias": bias.detach().tolist(), "objective": objective, "max_gradient": gradient,
            "converged": gradient <= 1e-5}


def predict(model, features):
    x = torch.as_tensor(features, dtype=torch.float64)
    z = (x - torch.tensor(model["mean"], dtype=torch.float64)) / torch.tensor(model["scale"], dtype=torch.float64)
    logits = torch.einsum("nd,dkc->nkc", z, torch.tensor(model["weights"], dtype=torch.float64))
    return (logits + torch.tensor(model["bias"], dtype=torch.float64)).log_softmax(-1)


def metrics(log_probs, targets):
    y = torch.as_tensor(targets, dtype=torch.long)
    correct = log_probs.argmax(-1) == y
    ce = -log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return {"accuracy": float(correct.double().mean()), "position_accuracy": correct.double().mean(0).tolist(),
            "cross_entropy": float(ce.mean()), "all_eight_correct": float(correct.all(1).double().mean()),
            "history_accuracy": correct.double().mean(1).tolist(), "history_ce": ce.mean(1).tolist()}


def room_counts(targets):
    labels = torch.as_tensor(targets, dtype=torch.long)
    result = torch.zeros(len(labels), 16, dtype=torch.float64)
    result[:, :6] = F.one_hot(labels, num_classes=6).sum(1).double() / 8
    return result


def positive_control():
    def examples(seed):
        labels = torch.randint(6, (512, 8), generator=torch.Generator().manual_seed(seed))
        angle = labels.double() * (2 * torch.pi / 6)
        return torch.stack((angle.cos(), angle.sin()), dim=-1).reshape(512, 16), labels

    train_x, train_y = examples(7331)
    test_x, test_y = examples(911)
    model = fit_heads(train_x, train_y)
    result = metrics(predict(model, test_x), test_y)
    return {"model": model, "metrics": result,
            "passed": model["converged"] and result["accuracy"] == 1.0}


def analyze_export(validated):
    meta, states, seal = validated
    labels = {split: torch.tensor([r["targets"] for r in meta["rows"][split]]) for split in states}
    model = fit_heads(states["train"], labels["train"])
    count_model = fit_heads(room_counts(labels["train"]), labels["train"])
    priors = F.one_hot(labels["train"], 6).sum(0).double() + 1
    priors = (priors / priors.sum(-1, keepdim=True)).log()
    result = {"arm": meta["arm"], "seed": meta["seed"], "export_seal": seal,
              "models": {"state": model, "room_counts": count_model}, "metrics": {},
              "train_state_std": states["train"].double().std(0, correction=0).tolist()}
    for split in ("train", "development"):
        identities = [r["history_id"] for r in meta["rows"][split]]
        donors = state_donors(identities)
        donor_rows = [identities.index(donors[identity]) for identity in identities]
        inputs = states[split]
        result["metrics"][split] = {
            "normal": metrics(predict(model, inputs), labels[split]),
            "shuffled": metrics(predict(model, inputs[donor_rows]), labels[split]),
            "room_counts": metrics(predict(count_model, room_counts(labels[split])), labels[split]),
            "position_only": metrics(priors.expand(len(inputs), -1, -1), labels[split]),
        }
    result["history_ids"] = {split: [r["history_id"] for r in meta["rows"][split]] for split in states}
    return result


def report(directories, source_root, declaration_path, tokenizer_dir):
    validated = validate_exports(directories, source_root, declaration_path, tokenizer_dir)
    control = positive_control()
    if not control["passed"]:
        return {"status": "positive_control_failed", "positive_control": control}
    runs = [analyze_export(data) for data in validated]
    expected = {(arm, seed) for arm in ("affine", "gelu") for seed in (1337, 2027, 4099)}
    if len(runs) != 6 or {(r["arm"], r["seed"]) for r in runs} != expected:
        raise ValueError("all six distinct checkpoints are required")
    if any(r["history_ids"] != runs[0]["history_ids"] for r in runs):
        raise ValueError("history order differs across exports")
    if any(not model["converged"] for r in runs for model in r["models"].values()):
        return {"status": "fit_not_converged", "positive_control": control, "runs": runs}
    count = len(runs[0]["history_ids"]["development"])
    weights = np.random.default_rng(0).multinomial(count, np.full(count, 1 / count), size=2000)
    contrasts, gates = {}, {}
    for arm in ("affine", "gelu"):
        arm_runs = sorted((r for r in runs if r["arm"] == arm), key=lambda r: r["seed"])
        contrasts[arm] = {}
        for condition in ("shuffled", "room_counts"):
            differences = np.asarray([
                np.asarray(r["metrics"]["development"]["normal"]["history_accuracy"])
                - r["metrics"]["development"][condition]["history_accuracy"] for r in arm_runs])
            means = differences.mean(1)
            interval = np.quantile((differences @ weights.T / count).mean(0), [0.025, 0.975])
            contrasts[arm][condition] = {"mean_difference": float(means.mean()),
                "seed_differences": {str(r["seed"]): float(v) for r, v in zip(arm_runs, means)},
                "seed_sd": float(means.std(ddof=1)), "interval95": interval.tolist()}
        gates[arm] = all(c["interval95"][0] > 0 and min(c["seed_differences"].values()) > 0
                          for c in contrasts[arm].values())
    return {"status": "complete", "positive_control": control, "runs": runs, "contrasts": contrasts,
            "binding_followup_gate": gates, "scope": "exploratory_privileged_ordered_content_not_entity_binding"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = report(args.exports, args.source_root, args.declaration, args.tokenizer)
    result["declaration_sha256"] = file_sha256(args.declaration)
    result["script_sha256"] = file_sha256(Path(__file__))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
