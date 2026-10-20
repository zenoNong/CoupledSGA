import os
import argparse
import warnings
from tqdm import tqdm


def dump_frames(args):
    video_dir = args.video_dir
    frame_dir = args.frame_dir
    annotation_dir = args.annotation_dir
    all_frames = args.all_frames

    # Load the list of annotated frames
    frame_list = []
    with open(os.path.join(annotation_dir, 'frame_list.txt'), 'r') as f:
        for frame in f:
            frame_list.append(frame.rstrip('\n'))

    # Create video to frames mapping
    video2frames = {}
    for path in frame_list:
        video, frame = path.split('/')
        if video not in video2frames:
            video2frames[video] = []
        video2frames[video].append(frame)

    # For each video, dump frames.
    for v in tqdm(video2frames):
        curr_frame_dir = os.path.join(frame_dir, v)
        if not os.path.exists(curr_frame_dir):
            video_path = os.path.join(video_dir, v)
            if not os.path.exists(video_path):
                warnings.warn(f'Video file {video_path} does not exist. Skipping.', RuntimeWarning)
                continue
            os.makedirs(curr_frame_dir, exist_ok=True)
            # Use ffmpeg to extract frames.
            try:
                import imageio_ffmpeg
                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg_bin = "ffmpeg"

            out_pattern = os.path.join(os.path.abspath(curr_frame_dir), "%06d.png")
            import subprocess
            cmd = [ffmpeg_bin, "-loglevel", "panic", "-y", "-i", os.path.abspath(video_path), out_pattern]
            subprocess.run(cmd)

            # if not keeping all frames, only keep the annotated frames included in frame_list.txt
            if not all_frames:
                keep_frames = set(video2frames[v])
                for frame in os.listdir(curr_frame_dir):
                    if frame not in keep_frames:
                        try:
                            os.remove(os.path.join(curr_frame_dir, frame))
                        except OSError:
                            pass
        else:
            warnings.warn('Frame directory %s already exists. Skipping dumping into this directory.' % curr_frame_dir,
                          RuntimeWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump frames")
    parser.add_argument("--video_dir", default="datasets/ag/videos",
                        help="Folder containing Charades videos.")
    parser.add_argument("--frame_dir", default="datasets/ag/frames",
                        help="Root folder containing frames to be dumped.")
    parser.add_argument("--annotation_dir", default="datasets/ag/annotations",
                        help=("Folder containing annotation files, including object_bbox_and_relationship.pkl, "
                              "person_bbox.pkl and frame_list.txt."))
    parser.add_argument("--all_frames", action="store_true",
                        help="Set if you want to dump all frames, rather than the frames listed in frame_list.txt")
    args = parser.parse_args()
    dump_frames(args)