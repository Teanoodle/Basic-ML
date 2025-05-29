import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator, TransformerMixin

class AttributesAdder(BaseEstimator, TransformerMixin):
    """特征添加器测试专用类"""
    def __init__(self):
        self.is_fitted_ = False
        
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        X = X.copy()
        for i in range(1, 6):
            X[f'newclose_{i}day'] = X['close'].shift(i)
        return X.dropna()

def test_feature_addition():
    """测试特征添加功能"""
    df = pd.DataFrame({
        'close': [10, 20, 30, 40, 50, 60, 70, 80]
    })
    expected = pd.DataFrame({
        'close': [60, 70, 80],
        'newclose_1day': [50.0, 60.0, 70.0],  # 改为float类型
        'newclose_2day': [40.0, 50.0, 60.0],
        'newclose_3day': [30.0, 40.0, 50.0],
        'newclose_4day': [20.0, 30.0, 40.0],
        'newclose_5day': [10.0, 20.0, 30.0]
    }, index=[5, 6, 7])  # 保留原始索引
    
    adder = AttributesAdder()
    adder.fit(df)  # 显式调用fit
    result = adder.transform(df)
    
    # 验证特征添加
    assert_frame_equal(result, expected)
    # 验证fit标志
    assert adder.is_fitted_ == True

def test_empty_input():
    """测试空输入处理"""
    df = pd.DataFrame({'close': []})
    adder = AttributesAdder()
    result = adder.transform(df)
    assert result.empty

def test_short_series():
    """测试短序列处理"""
    df = pd.DataFrame({'close': [10, 20]})  # 不足5天的数据
    adder = AttributesAdder()
    result = adder.transform(df)
    assert result.empty  # 应该返回空DataFrame

if __name__ == '__main__':
    test_feature_addition()
    test_empty_input()
    test_short_series()
    print("所有AttributesAdder测试通过！")
