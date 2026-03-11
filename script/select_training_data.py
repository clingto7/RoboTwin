import argparse
import csv
from pathlib import Path

import torch

from data_selection.feature_selector import (
    METRIC_REGISTRY,
    SELECTOR_REGISTRY,
    build_episode_embeddings,
    compute_subset_metrics,
    save_selection_result,
    select_episode_indices,
)


def _episode_files(
    task_name: str, task_config: str, episode_dir: Path | None
) -> list[Path]:
    if episode_dir is None:
        episode_dir = Path("data") / task_name / task_config / "data"
    files = sorted(
        episode_dir.glob("episode*.hdf5"),
        key=lambda p: int(p.stem.replace("episode", "")),
    )
    if not files:
        raise FileNotFoundError(f"No episode hdf5 found in {episode_dir}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select training episodes using visual-feature baseline"
    )
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("n_select", type=int)
    parser.add_argument("--episode-dir", type=Path, default=None)
    parser.add_argument("--model", type=str, default="dinov2_vits14")
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument(
        "--strategy",
        type=str,
        default="greedy_maxdist",
        choices=list(SELECTOR_REGISTRY.keys()),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine_distance",
        choices=list(METRIC_REGISTRY.keys()),
    )
    parser.add_argument("--n-subsets", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data_selection_outputs")
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    files = _episode_files(args.task_name, args.task_config, args.episode_dir)
    emb_res = build_episode_embeddings(
        episode_files=files,
        model_name=args.model,
        device=torch.device(args.device),
        sample_frames=args.sample_frames,
    )
    full_metrics = compute_subset_metrics(emb_res.embeddings)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    best_path: Path | None = None
    best_score = float("-inf")

    for i in range(args.n_subsets):
        run_seed = args.seed + i
        selected_ids, selected_metrics = select_episode_indices(
            episode_ids=emb_res.episode_ids,
            embeddings=emb_res.embeddings,
            strategy=args.strategy,
            n_select=args.n_select,
            seed=run_seed,
        )

        score = selected_metrics[args.metric]
        out_json = args.output_dir / f"selection_{args.strategy}_seed{run_seed}.json"
        save_selection_result(
            out_path=out_json,
            task_name=args.task_name,
            task_config=args.task_config,
            strategy=args.strategy,
            metric=args.metric,
            selected_ids=selected_ids,
            selected_metrics=selected_metrics,
            full_metrics=full_metrics,
        )

        rows.append(
            {
                "selection_file": str(out_json),
                "seed": run_seed,
                "strategy": args.strategy,
                "primary_metric": args.metric,
                "primary_metric_value": score,
                "cosine_distance": selected_metrics["cosine_distance"],
                "l2_distance": selected_metrics["l2_distance"],
                "variance_score": selected_metrics["variance_score"],
                "selected_episode_ids": ",".join(map(str, selected_ids)),
            }
        )

        if score > best_score:
            best_score = score
            best_path = out_json

    csv_path = args.output_dir / "selection_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "selection_file",
            "seed",
            "strategy",
            "primary_metric",
            "primary_metric_value",
            "cosine_distance",
            "l2_distance",
            "variance_score",
            "selected_episode_ids",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if best_path is not None:
        (args.output_dir / "best_selection.txt").write_text(
            str(best_path), encoding="utf-8"
        )
        print(f"Best selection: {best_path}")
    print(f"Summary: {csv_path}")


if __name__ == "__main__":
    main()
