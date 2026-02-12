import argparse
import glob
import json
from pathlib import Path

from tqdm import tqdm

from filter_one_wet import process_one_wet


def aggregate_stats(stats_files: list[str], out_path: str) -> None:
    totals = {}
    keys_sum = [
        "records_total", "records_with_text",
        "keep_lang", "keep_harmful", "keep_gopher", "keep_quality", "written",
        "masked_emails", "masked_phones", "masked_ips",        
    ]

    for k in keys_sum:
        totals[k] = 0
    
    per_file = []
    for sf in stats_files:
        with open(sf, "r", encoding="utf-8") as f:
            st = json.load(f)
        per_file.append(st)
        for k in keys_sum:
            totals[k] += int(st.get(k, 0))

    out = {"totals": totals, "files": per_file}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/data/CC/CC*.warc.wet.gz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stats-dir", required=True)
    ap.add_argument("--summary", default="runs/filter_summary.json")

    ap.add_argument("--max-files", type=int, default=100)
    ap.add_argument("--mode", choices=["local", "slurm"], default="local")

    # shared thresholds
    ap.add_argument("--lang-thr", type=float, default=0.8)
    ap.add_argument("--toxic-thr", type=float, default=0.8)
    ap.add_argument("--nsfw-thr", type=float, default=0.8)
    ap.add_argument("--quality-thr", type=float, default=0.5)
    ap.add_argument("--min-chars", type=int, default=200)

    # local mode
    ap.add_argument("--workers", type=int, default=8)

    # slurm mode (submitit)
    ap.add_argument("--slurm-parallel", type=int, default=16)
    ap.add_argument("--timeout-min", type=int, default=30)
    ap.add_argument("--mem-gb", type=int, default=2)
    ap.add_argument("--cpus-per-task", type=int, default=2)
    ap.add_argument("--partition", default="a4-cpu")
    ap.add_argument("--qos", default="a4-cpu-qos")
    ap.add_argument("--account", default="student")
    ap.add_argument("--logdir", default="slurm_logs")
    ap.add_argument("--dump-removed", type=int, default=0,
                    help="Number of removed/modified samples to dump (for inspection)")
    ap.add_argument("--removed-out", type=str, default="runs/removed_5.jsonl",
                    help="Where to write dumped removed/modified samples (jsonl)")
    ap.add_argument("--seed", type=int, default=0)    
    args = ap.parse_args()

    wet_paths = sorted(glob.glob(args.glob))[: args.max_files]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = Path(args.stats_dir); stats_dir.mkdir(parents=True, exist_ok=True)

    def _paths_for(in_path: str, out_dir: Path, stats_dir: Path) -> tuple[str, str]:
        name = Path(in_path).name
        out_path = str(out_dir / f"{name}.jsonl.gz")
        stats_path = str(stats_dir / f"{name}.stats.json")
        return out_path, stats_path

    stats_files: list[str] = []

    if args.mode == "local":
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = []
            for p in wet_paths:
                out_path, stats_path = _paths_for(p, out_dir, stats_dir)
                futs.append(
                    ex.submit(
                        process_one_wet,
                        p,
                        out_path,
                        stats_path,
                        lang_thr=args.lang_thr,
                        toxic_thr=args.toxic_thr,
                        nsfw_thr=args.nsfw_thr,
                        quality_thr=args.quality_thr,
                        min_chars=args.min_chars,
                        dump_removed=args.dump_removed,
                        removed_out=args.removed_out,
                        seed=args.seed

                    )
                )

            for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs)):
                _, stats_path = fut.result()
                stats_files.append(stats_path)

    else:
        import submitit

        executor = submitit.AutoExecutor(folder=args.logdir)
        executor.update_parameters(
            slurm_array_parallelism=args.slurm_parallel,
            timeout_min=args.timeout_min,
            mem_gb=args.mem_gb,
            cpus_per_task=args.cpus_per_task,
            slurm_account=args.account,
            slurm_partition=args.partition,
            slurm_qos=args.qos,
        )

        jobs = []
        with executor.batch():
            for p in wet_paths:
                out_path, stats_path = _paths_for(p, out_dir, stats_dir)
                jobs.append(
                    executor.submit(
                        process_one_wet,
                        p,
                        out_path,
                        stats_path,
                        lang_thr=args.lang_thr,
                        toxic_thr=args.toxic_thr,
                        nsfw_thr=args.nsfw_thr,
                        quality_thr=args.quality_thr,
                        min_chars=args.min_chars,
                        dump_removed=args.dump_removed,
                        removed_out=args.removed_out,
                        seed=args.seed                        
                    )
                )

        for j in tqdm(submitit.helpers.as_completed(jobs), total=len(jobs)):
            _, stats_path = j.result()
            stats_files.append(stats_path)


    aggregate_stats(stats_files, args.summary)
    print(f"Wrote summary to: {args.summary}")


if __name__ == "__main__":
    main()
