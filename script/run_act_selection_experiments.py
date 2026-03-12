import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], workdir: Path) -> None:
    subprocess.run(cmd, cwd=workdir, check=True)


def newest_eval_dir(task_name: str, task_config: str, ckpt_setting: str) -> Path | None:
    base = ROOT / "eval_result" / task_name / "ACT" / task_config / ckpt_setting
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def load_success_rate(result_file: Path) -> float:
    text = result_file.read_text(encoding="utf-8").strip().splitlines()
    if not text:
        return 0.0
    return float(text[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ACT experiments across selected subsets"
    )
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("n_select", type=int)
    parser.add_argument("n_subsets", type=int)
    parser.add_argument("epochs", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("gpu_id", type=int)
    parser.add_argument("--model", type=str, default="dinov2_vits14")
    parser.add_argument("--strategy", type=str, default="random")
    parser.add_argument("--metric", type=str, default="cosine_distance")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/data_selection")
    )
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selector_dir = output_dir / "selections"
    run(
        [
            "python",
            "script/select_training_data.py",
            args.task_name,
            args.task_config,
            str(args.n_select),
            "--model",
            args.model,
            "--strategy",
            args.strategy,
            "--metric",
            args.metric,
            "--n-subsets",
            str(args.n_subsets),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(selector_dir),
        ],
        ROOT,
    )

    summary_csv = selector_dir / "selection_summary.csv"
    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))

    exp_rows = []
    for row in rows:
        selection_file = Path(row["selection_file"])
        subset_seed = int(row["seed"])
        subset_tag = f"sel{subset_seed}"

        run(
            [
                "bash",
                "process_data.sh",
                args.task_name,
                args.task_config,
                str(args.n_select),
                str(selection_file),
                subset_tag,
            ],
            ROOT / "policy" / "ACT",
        )

        dataset_key = (
            f"sim-{args.task_name}-{args.task_config}-{args.n_select}-{subset_tag}"
        )
        ckpt_dir = (
            ROOT
            / "policy"
            / "ACT"
            / "act_ckpt"
            / f"act-{args.task_name}"
            / f"{args.task_config}-{args.n_select}-{subset_tag}"
        )
        run(
            [
                "python",
                "imitate_episodes.py",
                "--task_name",
                dataset_key,
                "--ckpt_dir",
                str(ckpt_dir),
                "--policy_class",
                "ACT",
                "--kl_weight",
                "10",
                "--chunk_size",
                "50",
                "--hidden_dim",
                "512",
                "--batch_size",
                "4",
                "--dim_feedforward",
                "3200",
                "--num_epochs",
                str(args.epochs),
                "--lr",
                "1e-5",
                "--save_freq",
                str(max(5, args.epochs // 2)),
                "--state_dim",
                "14",
                "--seed",
                str(args.seed),
            ],
            ROOT / "policy" / "ACT",
        )

        ckpt_setting = f"{args.task_config}-{subset_tag}"
        run(
            [
                "python",
                "script/eval_policy.py",
                "--config",
                "policy/ACT/deploy_policy.yml",
                "--overrides",
                "--task_name",
                args.task_name,
                "--task_config",
                args.task_config,
                "--ckpt_setting",
                ckpt_setting,
                "--ckpt_dir",
                str(ckpt_dir),
                "--seed",
                str(args.seed),
                "--temporal_agg",
                "true",
            ],
            ROOT,
        )

        eval_dir = newest_eval_dir(args.task_name, args.task_config, ckpt_setting)
        result_file = eval_dir / "_result.txt" if eval_dir is not None else None
        success_rate = (
            load_success_rate(result_file)
            if result_file and result_file.exists()
            else 0.0
        )

        exp_rows.append(
            {
                "subset_seed": subset_seed,
                "subset_tag": subset_tag,
                "selection_file": str(selection_file),
                "primary_metric": row["primary_metric"],
                "primary_metric_value": row["primary_metric_value"],
                "cosine_distance": row["cosine_distance"],
                "l2_distance": row["l2_distance"],
                "variance_score": row["variance_score"],
                "selected_episode_ids": row["selected_episode_ids"],
                "dataset_key": dataset_key,
                "ckpt_dir": str(ckpt_dir),
                "eval_dir": str(eval_dir) if eval_dir else "",
                "result_file": str(result_file) if result_file else "",
                "success_rate": success_rate,
            }
        )

    out_csv = output_dir / "experiment_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(exp_rows[0].keys()) if exp_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(exp_rows)

    if exp_rows:
        best = max(exp_rows, key=lambda x: x["success_rate"])
        (output_dir / "best_model.json").write_text(
            json.dumps(best, indent=2), encoding="utf-8"
        )

    print(f"Experiment results saved: {out_csv}")


if __name__ == "__main__":
    main()
