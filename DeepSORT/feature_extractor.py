"""切り出し画像からReID特徴量を抽出します。"""
import cv2
import numpy as np
import torch
import torchreid

# cv2: 画像のサイズ・色順の変換、numpy: 画像や特徴量を数値配列で扱う。
# torch: ニューラルネットワークの計算、torchreid: ReID用モデルの構築。
# ReID（再識別）は「別の時刻の対象が同じ個体か」を外見から判断する技術。
# このクラスはID自体を決めず、比較に使う数値の並び（特徴ベクトル）を返す。
# IDの対応付けはtracking.pyが担当する。

class TorchreidExtractor:
    # 初期化は最初の1回だけ。重みを毎フレーム読み直すと非常に遅くなるため。
    # selfは、このExtractorが保持するモデル・設定を参照するためのもの。
    def __init__(self, model_name, model_path, device="cuda",
                 num_classes=33, input_size=(256, 256), l2_normalize=True):
        self.device = device
        self.size = input_size
        self.l2_normalize = l2_normalize
        # まずネットワークの構造（計算のつながり）を作る。
        # num_classesは学習時の分類器の出力数。追跡できるID数の上限ではない。
        # 保存した重みの形と合わせるため、学習時と同じクラス数が必要。
        # pretrained=False: この後に自分の学習済み重みを読み込むため、
        # 別の事前学習済み重みをここで取得する必要はない。
        self.model = torchreid.models.build_model(
            name=model_name, num_classes=num_classes, pretrained=False
        )
        # checkpointは学習結果の保存データ。重み以外にepochなどを含む場合もある。
        # map_locationで読み込み先を指定し、CPU実行時にGPUへの読み込みを避ける。
        # weights_only=Falseは付随データも読む指定。信頼できる重みファイルを使う。
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # state_dictは「層の名前 → 学習した数値」の辞書。
        # 辞書の中に格納された形式と、重みの辞書だけを保存した形式の両方に対応。
        state_dict = checkpoint.get("state_dict", checkpoint)
        # 複数GPUで学習すると層名にmodule.が付くことがある。
        # 今の単体モデルの層名と一致させるため、先頭のmodule.だけ取り除く。
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        # 作った構造へ学習済みの数値をセットする。形や層名の不一致はエラーになる。
        self.model.load_state_dict(state_dict)
        # モデルと入力画像の計算場所は揃える必要がある（CPUまたはCUDA対応GPU）。
        self.model.to(device)
        # 学習モードから推論モードへ変更する。
        # BatchNormなどを推論時の動作にし、OSNetは分類結果ではなく特徴を返す。
        # eval()だけでは勾配計算は止まらないので、下でno_grad()も使う。
        self.model.eval()
        # 学習時と同じImageNetの平均・標準偏差で画像を標準化するための値。
        # RGBの順。float32はモデルの計算に合わせた32ビット小数型。
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, image):
        # 検出枠のサイズは対象ごとに違うため、学習時の入力サイズへ統一する。
        image = cv2.resize(image, self.size)
        # OpenCV画像はBGR順だが、モデルにはRGB順で渡す。
        # 疑似RGBの場合は[現在,中間,過去]→[過去,中間,現在]となり、学習条件に一致。
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 0～255の整数から0～1の小数へ。画像の値の範囲を学習時に合わせる。
        image = image.astype(np.float32) / 255.0
        # 各チャンネルで平均を引き、標準偏差で割る。これは画素値の標準化で、
        # 推論後の特徴ベクトルに行うL2正規化とは別の処理。
        image = (image - self.mean) / self.std
        # OpenCVの[高さ,幅,チャンネル]をPyTorchの[チャンネル,高さ,幅]へ並べ替える。
        # 縦横を回転する操作ではなく、配列の軸の順序を変える操作。
        return image.transpose(2, 0, 1)

    def __call__(self, crops):
        # __call__があるので、extractor(crops)という関数のような書き方で呼べる。
        # cropsは検出対象ごとの切り出し画像のリスト。
        # 検出0件ならモデルを呼ばず、特徴も0件で返す。
        if not crops:
            return np.empty((0, 0), dtype=np.float32)

        # 各画像を前処理してまとめる。N件なら形は[N,3,高さ,幅]。
        # まとめて推論することで、対象ごとにモデルを呼ぶ手間を減らす。
        batch = np.stack([self._preprocess(crop) for crop in crops])
        # NumPy配列をPyTorchのTensor（モデルが扱う配列）へ変換し、計算場所へ移す。
        inputs = torch.from_numpy(batch).to(self.device)
        # 推論だけなので、学習用の勾配（重みをどう修正するか）の記録は不要。
        # 記録を止めることでメモリと計算の無駄を減らす。
        with torch.no_grad():
            # モデル出力は[N,D]（Dは特徴の次元数）。OSNet x1_0では通常D=512。
            # DeepSORTがNumPy配列を使うため、CPUへ戻してNumPy形式に変換する。
            features = self.model(inputs).cpu().numpy()

        # L2正規化は各ベクトルをその長さで割り、長さをほぼ1に揃える処理。
        # 例：[3,4]の長さは5なので[0.6,0.8]になる。方向を残し、大きさを揃える。
        # configのl2_normalize=Falseならこの処理を省略する。
        # cosine距離はDeepSORT内部でも正規化するが、euclidean距離はしない。
        if self.l2_normalize:
            # axis=1: 対象ごとに特徴の長さを計算。keepdims=True: [N,1]を保ち、
            # 各行をそれぞれの長さで割れるようにする。1e-8はゼロ除算を防ぐ小さな値。
            features = features / (
                np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
            )
        # 行の順序はcropsと同じ。この対応を保って検出位置と特徴を結び付ける。
        return features
