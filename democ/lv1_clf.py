import numpy as np
from sklearn.neural_network import MLPClassifier


class LV1UserDefinedClassifierMLP1000HiddenLayerCorrectLabels:
    def __init__(self):
        self.clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=1000, activation='relu', learning_rate="invscaling")
        self.sampled_features = None
        self.sampled_labels = None

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):
        self.sampled_features = features
        self.sampled_labels = labels
        self.clf.fit(features, labels)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict(self, features):
        labels = self.clf.predict(features)
        return np.int32(self.correct_labels(features=features, labels=labels))

    @staticmethod
    def convert_to_px_arr(arr):
        image_size = 512
        return np.int32((arr + 1.0) * 0.5 * image_size)

    # サンプリング済み点は正解を返す
    def correct_labels(self, features, labels):
        features_px = self.convert_to_px_arr(features)
        sampled_features_px = self.convert_to_px_arr(self.sampled_features)

        features_px[:, 0] = features_px[:, 0] * 1000
        sampled_features_px[:, 0] = sampled_features_px[:, 0] * 1000

        features_px = np.sum(features_px, axis=1)
        sampled_features_px = np.sum(sampled_features_px, axis=1)

        for i, sampled in enumerate(sampled_features_px):
            index_list = np.where(sampled == features_px)[0]
            if len(index_list) > 0:
                labels[index_list[0]] = self.sampled_labels[i]

        return labels


class LV1UserDefinedClassifierMLP1700HiddenLayerCorrectLabels:
    def __init__(self):
        self.clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=1700, activation='relu', learning_rate="constant")
        self.sampled_features = None
        self.sampled_labels = None

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):
        self.sampled_features = features
        self.sampled_labels = labels
        self.clf.fit(features, labels)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict(self, features):
        labels = self.clf.predict(features)
        return np.int32(self.correct_labels(features=features, labels=labels))

    @staticmethod
    def convert_to_px_arr(arr):
        image_size = 512
        return np.int32((arr + 1.0) * 0.5 * image_size)

    # サンプリング済み点は正解を返す
    def correct_labels(self, features, labels):
        features_px = self.convert_to_px_arr(features)
        sampled_features_px = self.convert_to_px_arr(self.sampled_features)

        features_px[:, 0] = features_px[:, 0] * 1000
        sampled_features_px[:, 0] = sampled_features_px[:, 0] * 1000

        features_px = np.sum(features_px, axis=1)
        sampled_features_px = np.sum(sampled_features_px, axis=1)

        for i, sampled in enumerate(sampled_features_px):
            index_list = np.where(sampled == features_px)[0]
            if len(index_list) > 0:
                labels[index_list[0]] = self.sampled_labels[i]

        return labels
