import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator, TransformerMixin

class Imputer(BaseEstimator, TransformerMixin):
    """自定义的数据预处理类"""
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X.interpolate(method='linear', limit_direction='both')

def test_basic_imputation():
    """测试基本缺失值填充"""
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, 7, 8],
        'C': [np.nan, 10, 11, 12]
    })
    expected = pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0],  # 改为float类型
        'B': [5.0, 6.0, 7.0, 8.0],
        'C': [10.0, 10.0, 11.0, 12.0]  # 第一个值用后值填充
    })
    result = Imputer().transform(df)
    assert_frame_equal(result, expected)

def test_all_nan_column():
    """测试全NaN列处理"""
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [np.nan, np.nan, np.nan, np.nan]
    })
    expected = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [np.nan, np.nan, np.nan, np.nan]  # 全NaN列保持不变
    })
    result = Imputer().transform(df)
    assert_frame_equal(result, expected)

def test_edge_imputation():
    """测试边界情况填充"""
    df = pd.DataFrame({
        'A': [np.nan, 2, np.nan, 4, np.nan],
        'B': [1, np.nan, 3, np.nan, 5]
    })
    expected = pd.DataFrame({
        'A': [2.0, 2.0, 3.0, 4.0, 4.0],  # 改为float类型
        'B': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    result = Imputer().transform(df)
    assert_frame_equal(result, expected)

if __name__ == '__main__':
    test_basic_imputation()
    test_all_nan_column()
    test_edge_imputation()
    print("所有Imputer测试通过!")
