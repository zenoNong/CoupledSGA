"""
Filter Action Genome annotations down to a chosen video subset (e.g. 2000 train + 600 test)
BEFORE running tools/dump_frames.py, so you only extract frames you'll actually use.

Run this from inside your `dataset/ag/annotations/` folder, or point --ann_dir at it.

Usage:
    python filter_action_genome_subset.py \
        --ann_dir /path/to/dataset/ag/annotations \
        --out_dir /path/to/dataset/ag/annotations_subset \
        --n_train 2000 --n_test 600 --seed 42

    # or, if you already have a specific video ID list:
    python filter_action_genome_subset.py \
        --ann_dir /path/to/dataset/ag/annotations \
        --out_dir /path/to/dataset/ag/annotations_subset \
        --video_id_file my_2600_video_ids.txt
"""

import argparse
import pickle
import random
import os


def load_frame_list(path):
    with open(path, "r") as f:
        # each line looks like: VIDEO_ID/000090.png
        return [line.strip() for line in f if line.strip()]


def video_id_from_frame_tag(tag):
    # tag format: "VIDEO_ID/FRAME_ID.png" -> "VIDEO_ID"
    return tag.split("/")[0]


def select_video_ids(frame_list, n_train, n_test, seed):
    """
    Action Genome doesn't split train/test in frame_list.txt itself (that split
    lives inside the pkl metadata['set'] field), so here we just gather ALL
    unique video ids present in frame_list.txt and sample n_train + n_test from
    them. We cross-check against the pkl's train/test tags afterward.
    """
    all_ids = sorted(set(video_id_from_frame_tag(t) for t in frame_list))
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    chosen = all_ids[: n_train + n_test]
    return set(chosen)


def filter_frame_list(frame_list, video_ids):
    return [t for t in frame_list if video_id_from_frame_tag(t) in video_ids]


def filter_pickle_by_video(pkl_path, video_ids):
    """
    Works for both person_bbox.pkl and object_bbox_and_relationship*.pkl —
    both are dicts keyed by 'VIDEO_ID/FRAME_ID.png'.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    filtered = {
        k: v for k, v in data.items()
        if video_id_from_frame_tag(k) in video_ids
    }
    return filtered


def report_train_test_counts(filtered_obj_pkl):
    """
    object_bbox_and_relationship.pkl entries carry metadata['set'] = 'train'/'test'
    per-frame. Use this to sanity check your actual train/test split after filtering,
    since video IDs alone don't tell you the split.
    """
    counts = {"train": set(), "test": set()}
    for frame_tag, obj_list in filtered_obj_pkl.items():
        vid = video_id_from_frame_tag(frame_tag)
        for obj in obj_list:
            split = obj.get("metadata", {}).get("set")
            if split in counts:
                counts[split].add(vid)
    return {k: len(v) for k, v in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", required=True, help="folder containing frame_list.txt, person_bbox.pkl, object_bbox_and_relationship*.pkl")
    ap.add_argument("--out_dir", required=True, help="where filtered files get written")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--video_id_file", default=None, help="optional: text file with one video ID per line, overrides n_train/n_test sampling")
    ap.add_argument("--rel_pkl_name", default="object_bbox_and_relationship.pkl",
                     help="use 'object_bbox_and_relationship_filtersmall.pkl' if that's the variant your model repo expects (STTran/SceneSayer often use this one)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    frame_list_path = os.path.join(args.ann_dir, "frame_list.txt")
    person_pkl_path = os.path.join(args.ann_dir, "person_bbox.pkl")
    rel_pkl_path = os.path.join(args.ann_dir, args.rel_pkl_name)

    print("Loading frame_list.txt ...")
    frame_list = load_frame_list(frame_list_path)
    print(f"  total frames listed: {len(frame_list)}")

    if args.video_id_file:
        with open(args.video_id_file) as f:
            video_ids = set(line.strip() for line in f if line.strip())
        print(f"Using provided video ID list: {len(video_ids)} videos")
    else:
        video_ids = select_video_ids(frame_list, args.n_train, args.n_test, args.seed)
        print(f"Randomly selected {len(video_ids)} video IDs (seed={args.seed})")
        with open(os.path.join(args.out_dir, "selected_video_ids.txt"), "w") as f:
            for vid in sorted(video_ids):
                f.write(vid + "\n")

    # 1. Filter frame_list.txt
    filtered_frames = filter_frame_list(frame_list, video_ids)
    with open(os.path.join(args.out_dir, "frame_list.txt"), "w") as f:
        for t in filtered_frames:
            f.write(t + "\n")
    print(f"Filtered frame_list.txt -> {len(filtered_frames)} frames "
          f"(from {len(frame_list)})")

    # 2. Filter person_bbox.pkl
    print("Filtering person_bbox.pkl ...")
    filtered_person = filter_pickle_by_video(person_pkl_path, video_ids)
    with open(os.path.join(args.out_dir, "person_bbox.pkl"), "wb") as f:
        pickle.dump(filtered_person, f)
    print(f"  kept {len(filtered_person)} / entries for selected videos")

    # 3. Filter object_bbox_and_relationship(.pkl / _filtersmall.pkl)
    print(f"Filtering {args.rel_pkl_name} ...")
    filtered_rel = filter_pickle_by_video(rel_pkl_path, video_ids)
    with open(os.path.join(args.out_dir, args.rel_pkl_name), "wb") as f:
        pickle.dump(filtered_rel, f)
    print(f"  kept {len(filtered_rel)} entries for selected videos")

    # Copy object_classes.txt and relationship_classes.txt if present
    import shutil
    for fname in ["object_classes.txt", "relationship_classes.txt"]:
        src = os.path.join(args.ann_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out_dir, fname))

    # 4. Sanity check: report actual train/test video counts post-filter
    split_counts = report_train_test_counts(filtered_rel)
    print("\nActual train/test video counts after filtering "
          "(from metadata['set'] inside the pkl, NOT just your sample target):")
    print(f"  train videos: {split_counts.get('train', 0)}")
    print(f"  test videos:  {split_counts.get('test', 0)}")
    print("\nIMPORTANT: if these numbers don't match your intended 2000/600 split,")
    print("it's because random sampling over ALL video ids doesn't respect AG's")
    print("original train/test partition. Re-run with --video_id_file pointing at")
    print("IDs drawn separately from AG's train list and test list if you need an")
    print("exact 2000-train / 600-test split rather than an approximate one.")

    print(f"\nDone. Filtered annotations written to: {args.out_dir}")
    print("Next: point tools/dump_frames.py at this filtered frame_list.txt")
    print("      to extract only the frames you need.")


if __name__ == "__main__":
    main()
