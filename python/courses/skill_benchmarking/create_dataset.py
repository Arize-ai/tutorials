"""Load the link requests into an Arize AX dataset (Lesson 2).

    uv run create_dataset.py                       # data/requests.csv, 3 reps of each request
    uv run create_dataset.py --append data/new_requests.csv   # add new requests later (Lesson 5)

Each request goes in the dataset `--reps` times, so every experiment runs it that many times.
Agent runs vary, so one run of a request proves very little.
"""

import argparse
import csv
from pathlib import Path

from harness.ax import find_dataset
from harness.config import COURSE_DIR, arize_client, settings


def load_rows(path: Path, reps: int) -> list[dict]:
    with open(path, newline="") as f:
        requests = list(csv.DictReader(f))
    return [{**row, "rep": str(rep)} for row in requests for rep in range(1, reps + 1)]


def main() -> None:
    s = settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=COURSE_DIR / "data" / "requests.csv")
    parser.add_argument("--append", type=Path, help="Add the requests in this CSV to the existing dataset")
    parser.add_argument("--reps", type=int, default=3, help="How many times each request appears (default 3)")
    parser.add_argument("--name", default=s.dataset_name, help=f"Dataset name (default {s.dataset_name!r})")
    args = parser.parse_args()

    client = arize_client()
    existing = find_dataset(client, s, args.name)
    if args.append:
        if existing is None:
            raise SystemExit(f"No dataset named {args.name!r} to append to.")
        rows = load_rows(args.append, args.reps)
        client.datasets.append_examples(dataset=existing.id, space=s.arize_space_id, examples=rows)
        print(f"Added {len(rows)} examples to {args.name!r}")
        return
    if existing is not None:
        raise SystemExit(f"A dataset named {args.name!r} already exists. Pass --name to create another.")
    rows = load_rows(args.file, args.reps)
    dataset = client.datasets.create(name=args.name, space=s.arize_space_id, examples=rows)
    print(f"Created dataset {args.name!r} with {len(rows)} examples ({dataset.id})")


if __name__ == "__main__":
    main()
