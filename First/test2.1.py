import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator, TransformerMixin

# 这是最初报错的版本（缺少__init__方法）
class AttributesAdder(BaseEstimator, TransformerMixin):
    """特征添加器测试专用类（原始报错版本）"""
    def fit(self, X, y=None):
        self.is_fitted_ = True  # 这里会报错，因为is_fitted_未初始化
        return self
        
    def transform(self, X):
        X = X.copy()
        for i in range(1, 6):
            X[f'newclose_{i}day'] = X['close'].shift(i)
        return X.dropna()

def test_feature_addition():
    """测试特征添加功能（原始报错版本）"""
    df = pd.DataFrame({
        'close': [10, 20, 30, 40, 50, 60, 70, 80]
    })
    expected = pd.DataFrame({
        'close': [60, 70, 80],
        'newclose_1day': [50, 60, 70],  # 原始int类型
        'newclose_2day': [40, 50, 60],
        'newclose_3day': [30, 40, 50],
        'newclose_4day': [20, 30, 40],
        'newclose_5day': [10, 20, 30]
    }, index=[5, 6, 7])
    
    adder = AttributesAdder()
    result = adder.transform(df)  # 这里会报AttributeError
    
    assert_frame_equal(result, expected)
    assert adder.is_fitted_ == True  # 这里会报错

if __name__ == '__main__':
    print("这是最初报错的代码版本")
    print("运行时会出现两个错误:")
    print("1. AttributeError: is_fitted_未初始化")
    print("2. AssertionError: 数据类型不匹配")
    test_feature_addition()
