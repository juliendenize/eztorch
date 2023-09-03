import argparse
import os
import re
import subprocess
from pathlib import Path

ffmpeg_duration_template = re.compile(r"time=\s*(\d+):(\d+):(\d+)\.(\d+)")


def get_video_duration(video_file):
    cmd = ["ffmpeg", "-i", str(video_file), "-f", "null", "-"]

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(video_file, err.output)
        return -1

    try:
        output_decoded = output.decode()
        result_all = ffmpeg_duration_template.findall(output_decoded)
    except Exception as err:
        print(video_file, err, "chose to carry on decoding video")
        return 1

    if result_all:
        result = result_all[-1]
        duration = (
            float(result[0]) * 60 * 60
            + float(result[1]) * 60
            + float(result[2])
            + float(result[3]) * (10 ** -len(result[3]))
        )
    else:
        duration = -1
    return duration


def has_video_stream(video_file):
    cmd = [
        "ffprobe",
        "-i",
        str(video_file),
        "-show_streams",
        "-select_streams",
        "v",
        "-loglevel",
        "error",
    ]

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(video_file, err.output)
        return False

    return output != ""


def is_video_empty(video_file):
    return get_video_duration(video_file) <= 0 or not has_video_stream(video_file)


def process(row, folder_path, output_path, args):
    classname = row[0]
    videoname = row[1]
    videostem = row[2]

    inname = folder_path / classname / videoname

    if is_video_empty(inname):
        print(f"{inname} is empty.")
        return False, f"{inname} is empty."

    output_folder = output_path / classname
    if os.path.isdir(output_folder) is False:
        try:
            os.mkdir(output_folder)
        except:
            print(f"{output_folder} can't be created.")

    if args.downscale:
        downscaled_cmd = f"-c:v libx264 -filter:v \"scale='if(gt(ih,iw),{args.downscale_size},trunc(oh*a/2)*2):if(gt(ih,iw),trunc(ow/a/2)*2,{args.downscale_size})'\" -c:a copy"
    else:
        downscaled_cmd = ""

    if args.frames:
        outfile = "%08d.jpg"

        outfolder = output_folder / videostem
        outfolder.mkdir(exist_ok=False, parents=False)

        outname = outfolder / outfile
        frames_cmd = f"-q:v {args.video_quality}"
    else:
        outname = output_folder / videoname
        frames_cmd = ""

    if args.fps > 0:
        fps_cmd = f"-r {args.fps}"
    else:
        fps_cmd = ""

    status = False
    inname = '"%s"' % inname
    outname = '"%s"' % outname

    command = f"ffmpeg -loglevel panic -i {inname} {downscaled_cmd} {frames_cmd} {fps_cmd} {outname}"

    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE).stdout
    except subprocess.CalledProcessError as err:
        print(inname, outname, status, err.output)
        if args.frames:
            os.rmdir(outfolder)
        return status, err.output

    status = os.path.exists(outname)
    return status, "Process"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video dataset.")
    parser.add_argument("--input-folder", type=str, help="Location raw dataset.")
    parser.add_argument("--output-folder", type=str, help="Location to output folder.")
    parser.add_argument(
        "--downscale", action="store_true", help="If True, downscale video."
    )
    parser.add_argument(
        "--downscale-size", type=int, default=256, help="Shorted side for downscale."
    )
    parser.add_argument("--frames", action="store_true", help="Extract frames")
    parser.add_argument(
        "--video-quality", type=int, default=1, help="Frames quality for ffmpeg."
    )
    parser.add_argument("--fps", type=int, default=-1, help="Fps to process the video.")

    args = parser.parse_args()

    num_tasks = int(os.environ["SLURM_NTASKS"]) if "SLURM_NTASKS" in os.environ else 1
    local_rank = int(os.environ["SLURM_PROCID"]) if "SLURM_PROCID" in os.environ else 0

    print(f"rank {local_rank} / {num_tasks} tasks")
    print("args:", "\n", args)

    folder_path = Path(args.input_folder)
    output_path = Path(args.output_folder)

    output_path.mkdir(exist_ok=True, parents=True)

    file_list = [
        [str(path.parent.name), str(path.name), str(path.stem)]
        for path in Path(folder_path).rglob("*")
        if path.is_file() and path.suffix != ".txt"
    ]

    print(f"Number of videos found in {folder_path} {len(file_list)}.")

    print(f"start processing from {folder_path} to {output_path}")
    for i, row in enumerate(file_list):
        if i % num_tasks == local_rank:
            process(row, folder_path, output_path, args)
    print("end processing")
