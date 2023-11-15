import argparse
import os
import time
from typing import Dict, Tuple

import cv2
import numpy as np

from detect import Detect
from streams import VideoStream
from tracker import Tracker
from utils import direction_config


def _detect_person(
    detect: Detect,
    frame: np.ndarray,
    confidence: float,
    iou_threshold: float,
) -> np.ndarray:
    """Mendeteksi objek orang dalam bingkai.

    Pengembalian:
        np.ndarray: Array seperti [xyxy, skor].
    """

    # Deteksi objek dalam bingkai.
    boxes, scores, class_idx = detect.detect(frame)

    # NMS(Non-Maximum Suppression)
    idx = cv2.dnn.NMSBoxes(boxes, scores, confidence, iou_threshold)
    boxes = boxes[idx]
    scores = scores[idx]
    class_idx = class_idx[idx]

    # Filter hanya objek orang (class index = 0).
    person_idx = np.where(class_idx == 0)[0]
    boxes = boxes[person_idx]
    scores = scores[person_idx]

    # Scale boxes berdasarkan ukuran bingkai.
    H, W = frame.shape[:2]
    boxes = detect.to_xyxy(boxes) * np.array([W, H, W, H])

    # dets:  [xmin, ymin, xmax, ymax, score]
    dets = np.concatenate([boxes.astype(int), scores.reshape(-1, 1)], axis=1)
    return dets


def main(
    src: str,
    dest: str,
    model: str,
    video_fmt: str,
    confidence: float,
    iou_threshold: float,
    directions: Dict[str, Tuple[bool]],
):
    """Lacak objek manusia dan hitung jumlah manusia.

    Argumen:
        src (str): Sumber video.
        dest (str): Direktori untuk menyimpan hasil.
        model (str): Jalur menuju bobot tflite.
        kepercayaan diri (float): Ambang batas kepercayaan.
        iou_threshold (float): Ambang batas IoU untuk NMS.
    """
    if not os.path.exists(dest):
        os.mkdir(dest)

    # The line to count.
    border = [(0, 500), (1920, 500)]
    directions = {key: direction_config.get(d_str) for key, d_str in directions.items()}
    tracker = Tracker(border, directions)
    detect = Detect(model, confidence)
    stream = VideoStream(src)
    writer = None

    total_frames = len(stream)
    if total_frames:
        print(f"Total frames: {len(stream)}")

    while True:
        # Baca frame berikutnya dari streaming.
        is_finish, frame = stream.next()

        if not is_finish:
            break

        start = time.time()
        dets = _detect_person(detect, frame, confidence, iou_threshold)
        end = time.time()

        # Perbarui pelacak dan gambar kotak pembatas dalam bingkai.
        # dets:  [xmin, ymin, xmax, ymax, score]
        frame = tracker.update(frame, dets)

        # Hanya dieksekusi pertama kali.
        if writer is None:
            # Initialize video writer.
            model_name = os.path.basename(model).split(".")[0]
            video_name = os.path.basename(src).split(".")[0]
            codecs = {"mp4": "MP4V", "avi": "MJPG"}
            basename = f"{video_name}_{model_name}"
            output_video = os.path.join(dest, f"{basename}.{video_fmt}")
            fourcc = cv2.VideoWriter_fourcc(*codecs[video_fmt])
            writer = cv2.VideoWriter(output_video, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

            # Estimasi total time.
            second_per_frame = end - start
            print(f"Waktu komputasi per frame: {second_per_frame:.4f} seconds")
            print(f"Perkiraan total waktu: {second_per_frame * total_frames:.4f}")

        # Save frame as an image and video.
        cv2.imwrite(os.path.join(dest, f"{basename}.jpg"), frame)
        writer.write(frame)

    writer.release()
    stream.release()
    print("Selesai!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Path to video source.", default="./data/TownCentreXVID.mp4")
    parser.add_argument("--dest", help="Path to output directory", default="./outputs/")
    parser.add_argument("--model", help="Path to YOLOv5 tflite file", default="./models/yolov5n6-fp16.tflite")
    parser.add_argument("--video-fmt", help="Format of output video file.", choices=["mp4", "avi"], default="mp4")
    parser.add_argument("--confidence", type=float, default=0.2, help="Confidence threshold.")
    parser.add_argument("--iou-threshold", type=float, default=0.2, help="IoU threshold for NMS.")
    parser.add_argument("--directions", default={"total": None}, type=eval, help="Directions")

    args = vars(parser.parse_args())
    main(**args)
