
import os
from argparse import ArgumentParser

import cv2


def _main(video: str, save_dir: str, limit_frames: int) -> None:
    """ "Convert video to images.

    Args:
        video(str): Jalur ke file video.
        save_dir(str): Direktori untuk menyimpan gambar.
        limit_frames(int): Jumlah maksimum frame yang akan disimpan.
    """
    if not os.path.exists(video):
        raise Exception("Video file not found.")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = 0

    # Video to images.
    while success and count < limit_frames:
        cv2.imwrite(os.path.join(save_dir, f"frame{str(count).zfill(5)}.jpg"), image)
        success, image = vidcap.read()
        print("Read a new frame: ", success)
        count += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--save-dir", default="./data/frames/")
    parser.add_argument("--limit-frames", default=800, type=int)
    args = parser.parse_args()
    _main(**vars(args))
