from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from sort import Sort
from utils import check_direction, is_intersect


class Tracker(object):
    def __init__(
        self,
        border: List[Tuple[int]],
        directions: Tuple[bool],
        count_callback: Optional[Callable] = None,
    ):
        """Constructor of Tracker.

        Args:
            border (List[Tuple[int]]): Perbatasan untuk mendeteksi jumlah.
            count_callback (Opsional[Callable], opsional): Fungsi panggilan balik yang akan dipanggil saat penghitung habis.
                                                           Ambil counter(int) untuk argumen.
        """
        self.tracker = Sort()
        self.border = border
        self.count_callback = count_callback
        self.memory = {}
        self.counter = {key: 0 for key in directions.keys()}
        self.directions = directions

        np.random.seed(2021)
        self.COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    def _is_count(
        self,
        center: Tuple[int],
        center_prev: Tuple[int],
        border: List[Tuple[int]],
        key: str,
    ) -> bool:
        """Periksa apakah dihitung atau tidak.

        1. check_direction : Mengecek arah pergerakan manusia.
                            Jika arah tidak ditentukan, kembalikan True.
        2. is_intersect: Periksa apakah perbatasan dan pergerakan manusia berpotongan.

        Argumen:
            center(Tuple[int]): Posisi tengah saat ini.
            center_prev(Tuple[int]): Posisi tengah sebelumnya.
            perbatasan(Daftar[Tuple[int]]): Perbatasan.
            kunci(str): "dalam", "luar" atau "total".
        """

        return check_direction(center, center_prev, self.directions[key]) and is_intersect(
            center, center_prev, border[0], border[1]
        )

    def update(self, frame: np.ndarray, dets: np.ndarray) -> np.ndarray:
        """Perbarui pelacak dan gambar kotak pembatas dalam bingkai.

        Arg:
            bingkai (np.ndarray): Bingkai target.
            dets (np.ndarray): Array seperti [xyxy + skor].

        Pengembalian:
            np.ndarray: Bingkai dengan kotak pembatas dan hitungan.
        """
        # Perbarui Sortir.
        tracks = self.tracker.update(dets)

        boxes = []
        index_ids = []
        previous = self.memory.copy()

        for track in tracks.astype(int):
            boxes.append([track[0], track[1], track[2], track[3]])
            index_ids.append(track[4])
            # Tambahkan id indeks dan kotak ke memori.
            self.memory[index_ids[-1]] = boxes[-1]

        if len(boxes) == 0:
            return frame

        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box

            color = [int(c) for c in self.COLORS[index_ids[i] % len(self.COLORS)]]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            if index_ids[i] in previous:
                previous_box = previous[index_ids[i]]
                xmin2, ymin2, wmax2, ymax2 = previous_box

                # Hitung pusat kotak pembatas.
                center = (int(xmin + (xmax - xmin) / 2), int(ymin + (ymax - ymin) / 2))
                center_prev = (int(xmin2 + (wmax2 - xmin2) / 2), int(ymin2 + (ymax2 - ymin2) / 2))

                # Gambarlah gerakan kotak pembatas.
                cv2.line(frame, center, center_prev, color, 3)

                callback = False
                for key in self.directions.keys():
                    if self._is_count(center, center_prev, self.border, key):
                        self.counter[key] += 1
                        callback = True

                # Execute callback.
                if self.count_callback and callback:
                    self.count_callback(self.counter)

            # Letakkan ID di kotak
            cv2.putText(
                frame,
                str(index_ids[i]),
                (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Draw border.
        cv2.line(frame, self.border[0], self.border[1], (10, 255, 0), 3)
        # Letakkan penghitung di sudut kiri atas.
        for i, (key, count) in enumerate(self.counter.items()):
            cv2.putText(
                frame,
                f"{key}: {count}",
                (30, 30 + 80 * (i + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (10, 255, 0),
                5,
            )
        return frame
