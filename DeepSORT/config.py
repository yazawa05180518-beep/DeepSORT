"""実行設定。相対パスはこのファイルのあるフォルダを基準にします。"""
from dataclasses import dataclass
from pathlib import Path


# 実行した場所に依存せず、この設定ファイルのあるフォルダからパスを作る。
BASE_DIR = Path(__file__).resolve().parent


# 設定値をまとめるクラス。dataclassが初期化処理を自動生成する。
@dataclass
class Config:
    # 対象を検出するYOLOの重みファイル。
    yolo_model_path: Path = BASE_DIR / "r7a.pt"
    # 外見の特徴を抽出するReIDの重み。通常画像／疑似RGBの学習条件に合わせる。
    reid_model_path: Path = BASE_DIR / "model.pth.tar-100"
    # 読み込む動画。
    video_input: Path = BASE_DIR / "test.MOV"
    # 枠とIDを描画した動画の保存先。同名ファイルがあれば上書きする。
    video_output: Path = BASE_DIR / "log/result_D.mp4"
    # 評価用の追跡座標データの保存先。
    result_txt: Path = BASE_DIR / "log/result_D.txt"

    # 重みを学習した時と同じモデル構造を指定する。
    reid_name: str = "osnet_x1_0"
    # 学習時のクラス数。重みの形を合わせるための値で、追跡ID数の制限ではない。
    reid_num_classes: int = 33
    reid_input_size: tuple = (256, 256)  # (幅, 高さ)。学習条件に合わせる

    # True: 抽出特徴をL2正規化、False: モデルの出力をそのまま渡す。
    # DeepSORTのcosine距離は内部で正規化するため、Falseでも距離計算時は
    # 正規化されます。画像のImageNet標準化とは別の設定です。
    l2_normalize: bool = True

    # YOLOの検出対象クラス。現在のr7a.ptでは1がboar（イノシシ）。
    class_ids: tuple = (1,)
    # これ未満の検出信頼度を除外する。大きくすると弱い検出を採用しにくい。
    confidence_threshold: float = 0.3
    # "cosine" または "euclidean"。euclideanは二乗ユークリッド距離です。
    distance_metric: str = "cosine"
    # 距離の種類・L2正規化の有無に合わせて調整してください。
    # 外見の距離の上限。小さいほど外見の対応付けが厳しくなる。
    matching_threshold: float = 0.5
    # 各IDが保持する特徴ベクトルの最大数。古い特徴から削除する。
    feature_budget: int = 300
    # IoU照合の上限は1-IoU。0.3なら重なりIoUが0.7以上必要。
    max_iou_distance: float = 0.3
    # 確定済みIDを内部で保持する未検出フレーム数。この値を超えると削除する。
    max_age: int = 300
    # この回数連続で検出が対応付いたらIDを確定する。未確定中の見失いは削除。
    n_init: int = 3
    # 未検出でも予測枠を描画・保存するフレーム数。内部保持のmax_ageとは別。
    output_max_age: int = 3
