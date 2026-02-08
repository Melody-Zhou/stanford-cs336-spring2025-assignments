import argparse
from pathlib import Path
from collections import Counter

import fasttext


def count_labels(train_path: Path) -> Counter:
    """Count fastText labels from a supervised training file."""
    c = Counter()
    with train_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # fastText supervised format: "__label__X <text...>"
            parts = line.split(maxsplit=1)
            if parts and parts[0].startswith("__label__"):
                c[parts[0]] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("runs/quality_train.txt"),
                    help="Path to fastText supervised training file")
    ap.add_argument("--out", type=Path, default=Path("runs/quality_fasttext.bin"),
                    help="Where to save the trained .bin model")
    # Common fastText supervised hyperparameters
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--epoch", type=int, default=10)
    ap.add_argument("--wordNgrams", type=int, default=2)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--minn", type=int, default=2)
    ap.add_argument("--maxn", type=int, default=5)
    ap.add_argument("--loss", type=str, default="softmax", choices=["softmax", "hs", "ova"])
    ap.add_argument("--thread", type=int, default=8)
    ap.add_argument("--verbose", type=int, default=2)
    args = ap.parse_args()

    if not args.train.exists():
        raise FileNotFoundError(f"Training file not found: {args.train}")

    label_counts = count_labels(args.train)
    total = sum(label_counts.values())
    print(f"Training file: {args.train}")
    print(f"Total samples: {total}")
    print(f"Label distribution: {dict(label_counts)}")

    # Train
    model = fasttext.train_supervised(
        input=str(args.train),
        lr=args.lr,
        epoch=args.epoch,
        wordNgrams=args.wordNgrams,
        dim=args.dim,
        minn=args.minn,
        maxn=args.maxn,
        loss=args.loss,
        thread=args.thread,
        verbose=args.verbose,
    )

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.out))
    print(f"Saved model to: {args.out}")

    # Quick sanity check on the training set (not a real evaluation)
    n_test, precision, recall = model.test(str(args.train))
    print(f"Train-set test: n={n_test} precision@1={precision:.4f} recall@1={recall:.4f}")


if __name__ == "__main__":
    main()
