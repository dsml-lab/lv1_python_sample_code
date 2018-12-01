import math

import numpy as np

from democ.distance import find_furthest_place
from democ.lv1_clf import LV1UserDefinedClassifierMLP1000HiddenLayerCorrectLabels, \
    LV1UserDefinedClassifierMLP1700HiddenLayerCorrectLabels
from democ.voter import Lv1Voter, Voter


class Parliament:
    """議会クラス"""

    @staticmethod
    def get_image_size(exe_n):
        """サンプリング候補点を生成する際に使われる2次元座標の幅"""
        return math.ceil(math.sqrt(exe_n)) + 128
        # return 512

    @staticmethod
    def get_samplable_features_2_dimension(image_size):
        """サンプリング候補点を生成する"""
        h = image_size // 2
        point_count = image_size * image_size
        samplable_features = np.zeros((point_count, 2))
        for i in range(0, point_count):
            x = i % image_size
            y = i // image_size
            samplable_features[i][0] = np.float32((x - h) / h)
            samplable_features[i][1] = np.float32(-(y - h) / h)
        return np.float32(samplable_features)

    @staticmethod
    def create_lv1_voters():
        """投票者を生成する"""
        voters = [
            Lv1Voter(model=LV1UserDefinedClassifierMLP1000HiddenLayerCorrectLabels(), label_size=10),
            Lv1Voter(model=LV1UserDefinedClassifierMLP1700HiddenLayerCorrectLabels(), label_size=10)]
        return voters

    def __init__(self, samplable_features, voter1: Voter, voter2: Voter):
        self.voter1 = voter1
        self.voter2 = voter2
        self.samplable_features = samplable_features  # サンプリング候補点

    def get_optimal_solution(self, sampled_features, sampled_likelihoods):
        """最適なサンプリング点を取得する"""
        self.__fit_to_voters(sampled_features=sampled_features, sampled_likelihoods=sampled_likelihoods)  # 投票者を訓練
        self.__predict_to_voters()  # 投票者による予測

        # # すべての投票者の投票結果を集計
        # 識別結果1と2の差分をとる
        samplable_likelihoods_diff = np.absolute(
            self.voter1.get_samplable_likelihoods() - self.voter2.get_samplable_likelihoods())

        # 同じ点の値を合計し、1次元行列に変換
        discrepancy_arr = samplable_likelihoods_diff.max(axis=1)

        max_value = np.amax(discrepancy_arr)
        index_list = np.where(discrepancy_arr == max_value)[0]
        filtered_samplable_features = self.samplable_features[index_list]

        opt_feature = find_furthest_place(sampled_features=sampled_features,
                                          filtered_samplable_features=filtered_samplable_features)

        self.delete_samplable_features(delete_feature=opt_feature)

        return opt_feature

    def delete_samplable_features(self, delete_feature):
        """サンプリング候補点からサンプリング済み点を消す"""
        index_list = np.where(delete_feature == self.samplable_features)[0]

        # サンプリング候補から除外
        self.samplable_features = np.delete(self.samplable_features, index_list[0], axis=0)

    def __fit_to_voters(self, sampled_features, sampled_likelihoods):
        """投票者を学習"""
        self.voter1.sampled_fit(sampled_features=sampled_features, sampled_likelihoods=sampled_likelihoods)
        self.voter2.sampled_fit(sampled_features=sampled_features, sampled_likelihoods=sampled_likelihoods)

    def __predict_to_voters(self):
        """投票者による予測"""
        self.voter1.samplable_predict(samplable_features=self.samplable_features)
        self.voter2.samplable_predict(samplable_features=self.samplable_features)
