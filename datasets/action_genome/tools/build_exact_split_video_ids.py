"""
Build an EXACT 2000-train / 600-test video ID list from Action Genome's real
train/test partition, before running filter_action_genome_subset.py.

Why this is needed: frame_list.txt has no train/test label. The real split
lives inside object_bbox_and_relationship.pkl, per-frame, under
metadata['set'] ('train' or 'test'). Random sampling over frame_list.txt
alone does NOT respect this partition. This script reads the split first,
then samples exactly the counts you want from each side separately.

Usage:
    python datasets/action_genome/tools/build_exact_split_video_ids.py \
        --ann_dir datasets/ag/annotations_original \
        --rel_pkl_name object_bbox_and_relationship.pkl \
        --n_train 45 --n_test 5 --seed 10 \
        --out video_50_file.txt

    python datasets/action_genome/tools/build_exact_split_video_ids.py --ann_dir datasets/ag/annotations_original --rel_pkl_name object_bbox_and_relationship.pkl --n_train 45 --n_test 5 --seed 10 --out video_50_file.txt

Then feed the output straight into:
    python filter_action_genome_subset.py \
        --ann_dir ... --out_dir ... --video_id_file video_id_file.txt
"""

import argparse
import pickle
import random
import os


def video_id_from_frame_tag(tag):
    return tag.split("/")[0]


def build_split_index(rel_pkl_path):
    """
    Returns: (train_ids, test_ids, mixed_ids) as sets of video IDs.
    A video is assigned to a split based on the 'set' value(s) seen across
    its frames (AG videos should be entirely one split; mixed-tag edge
    cases are excluded and reported).
    """
    with open(rel_pkl_path, "rb") as f:
        data = pickle.load(f)

    video_splits = {}  # video_id -> set of split labels seen
    for frame_tag, obj_list in data.items():
        vid = video_id_from_frame_tag(frame_tag)
        for obj in obj_list:
            split = obj.get("metadata", {}).get("set")
            if split is None:
                continue
            video_splits.setdefault(vid, set()).add(split)

    train_ids, test_ids, mixed_ids = set(), set(), set()
    for vid, splits in video_splits.items():
        if splits == {"train"}:
            train_ids.add(vid)
        elif splits == {"test"}:
            test_ids.add(vid)
        else:
            mixed_ids.add(vid)  # shouldn't normally happen in AG

    return train_ids, test_ids, mixed_ids


def sample_exact(ids, n, seed, label):
    ids = sorted(ids)  # sort first for reproducibility across platforms
    if len(ids) < n:
        raise ValueError(
            f"Requested {n} {label} videos but only {len(ids)} are available "
            f"in the full dataset. Lower --n_{label} or check your pkl path."
        )
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", required=True,
                     help="folder containing the ORIGINAL, unfiltered annotation pkl (full ~10K videos)")
    ap.add_argument("--rel_pkl_name", default="object_bbox_and_relationship.pkl",
                     help="use the exact filename your target repo expects, e.g. "
                          "object_bbox_and_relationship_filtersmall.pkl")
    ap.add_argument("--video_dir", default=None,
                     help="optional: directory containing actual video files (e.g. datasets/ag/videos) to filter against")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="video_id_file.txt")
    args = ap.parse_args()

    rel_pkl_path = os.path.join(args.ann_dir, args.rel_pkl_name)

    print(f"Reading {rel_pkl_path} (this is the FULL, unfiltered file — "
          f"do not point this at an already-subset pkl) ...")
    train_ids, test_ids, mixed_ids = build_split_index(rel_pkl_path)

    if args.video_dir and os.path.exists(args.video_dir):
        available_files = set(os.listdir(args.video_dir))
        train_ids = {v for v in train_ids if v in available_files or os.path.splitext(v)[0] in available_files or f"{v}.mp4" in available_files}
        test_ids = {v for v in test_ids if v in available_files or os.path.splitext(v)[0] in available_files or f"{v}.mp4" in available_files}
        print(f"\nFiltered against video directory '{args.video_dir}':")

    print(f"\nFull dataset split found:")
    print(f"  train videos available: {len(train_ids)}")
    print(f"  test videos available:  {len(test_ids)}")
    if mixed_ids:
        print(f"  WARNING: {len(mixed_ids)} video(s) had frames tagged with "
              f"more than one split label — excluded from sampling. "
              f"IDs: {sorted(mixed_ids)[:10]}{' ...' if len(mixed_ids) > 10 else ''}")

    chosen_train = sample_exact(train_ids, args.n_train, args.seed, "train")
    chosen_test = sample_exact(test_ids, args.n_test, args.seed, "test")

    with open(args.out, "w") as f:
        for vid in chosen_train + chosen_test:
            f.write(vid + "\n")

    # Also write separate lists — useful for double-checking your training
    # script actually respects this exact split rather than re-splitting itself.
    train_out = args.out.replace(".txt", "_train.txt")
    test_out = args.out.replace(".txt", "_test.txt")
    with open(train_out, "w") as f:
        f.write("\n".join(chosen_train) + "\n")
    with open(test_out, "w") as f:
        f.write("\n".join(chosen_test) + "\n")

    print(f"\nSampled exactly {len(chosen_train)} train + {len(chosen_test)} test videos.")
    print(f"Combined list written to: {args.out}")
    print(f"Train-only list written to: {train_out}")
    print(f"Test-only list written to: {test_out}")
    print("\nNext step: run filter_action_genome_subset.py with")
    print(f"  --video_id_file {args.out}")
    print("to produce the matching filtered frame_list.txt + pkls.")
    print("\nIMPORTANT: whatever model repo you train with (STTran/SceneSayer/etc.)")
    print("likely re-derives train/test membership itself from metadata['set']")
    print("inside the pkl at load time — since we only kept 'train'-tagged and")
    print("'test'-tagged videos respectively with no mixing, this will match.")


if __name__ == "__main__":
    main()
