import os
from glob import glob
from typing import Any, Tuple

import cv2
import imutils


class BaseStream(object):
    """Base streamer."""

    def next(self) -> Tuple[bool, Any]:
        """Read next frame."""
        raise NotImplementedError()

    def release(self) -> None:
        """Close stream."""
        raise NotImplementedError()


class VideoStream(BaseStream):
    """Video stream."""

    def __init__(self, file: str):
        self.stream = cv2.VideoCapture(file)

    def next(self) -> Tuple[bool, Any]:
        """Read next frame.

        Returns:
            if_finish(bool): Mengembalikan True jika tidak ada bingkai.
            frame(Apa saja): Kembalikan np.ndarray jika bingkai ada.
        """
        if_finish, frame = self.stream.read()
        return if_finish, frame

    def release(self):
        """Release video stream."""
        self.stream.release()

    def __len__(self):
        """Mengembalikan total frame."""
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
            total_frames = int(self.stream.get(prop))
            return total_frames
        # pylint: disable=bare-except
        except:
            return None


class ImageFileStream(BaseStream):
    """Image file stream.

    Args:
        src_dir(str): Jalur ke direktori yang berisi bingkai sebagai file jpg.
    """

    def __init__(self, src_dir: str):
        # Hasilkan daftar yang diurutkan.
        # Asumsikan frame dihasilkan oleh /data/video2img.py.
        self.images = sorted(glob(os.path.join(src_dir, "*.jpg")))
        self.index = 0

    def next(self) -> Tuple[bool, Any]:
        """Read next frame."""
        try:
            frame = cv2.imread(self.images[self.idx])
            self.index += 1
            return True, frame

        except (IndexError):
            return False, None

    def release(self):
        pass

    def __len__(self):
        return len(self.images)
