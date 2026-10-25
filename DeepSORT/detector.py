"""YOLOの検出結果を追跡処理に渡せる形式にまとめます。"""
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class DetectionResult:
    # 1件の検出をひとまとめにする入れ物。位置と画像の対応がずれないようにする。
    bbox: tuple  # (左上x, 左上y, 幅, 高さ)
    confidence: float
    class_name: str
    crop: np.ndarray


class YoloDetector:
    def __init__(self, model_path, class_ids=(1,), confidence_threshold=0.3):
        # 学習済みYOLOを読み込む。class_ids=Noneならクラスで絞り込まない。
        self.model = YOLO(str(model_path))
        self.class_ids = list(class_ids) if class_ids is not None else None
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        # 画像配列の先頭2軸は高さと幅。枠を画面内へ収めるために使用する。
        height, width = frame.shape[:2]
        # 1フレームの検出を実行。[0]は、この1枚に対する結果を取り出す指定。
        result = self.model(
            frame, classes=self.class_ids,
            conf=self.confidence_threshold, verbose=False
        )[0]
        detections = []
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < self.confidence_threshold:
                continue
            # YOLOの左上・右下座標を、画像の切り出しに使える整数へ変換する。
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 画面外の座標を切り詰め、空の領域はスキップする。
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            # DeepSORT向けに[左上x,左上y,幅,高さ]へ変換する。
            # NumPy画像の切り出しは[yの範囲,xの範囲]の順。
            detections.append(DetectionResult(
                bbox=(x1, y1, x2 - x1, y2 - y1),
                confidence=confidence,
                class_name=self.model.names[int(box.cls[0])],
                crop=frame[y1:y2, x1:x2],
            ))
        return detections
