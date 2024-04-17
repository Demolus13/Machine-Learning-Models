'''
Metrics For Machine Learning Models
'''

from typing import Union
import pandas as pd
import numpy as np

def Accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    y_hat: pd.Series: Predicted values
    y: pd.Series: Actual values

    Returns: float: Accuracy
    """
    assert y_hat.size == y.size
    accuracy = (y_hat == y).mean()
    return accuracy


def Precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    y_hat: pd.Series: Predicted values
    y: pd.Series: Actual values
    cls: Union[int, str]: Class to calculate precision for

    Returns: float: Precision
    """
    assert y_hat.size == y.size
    tp = ((y_hat == cls) & (y == cls)).sum()
    fp = ((y_hat == cls) & (y != cls)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return precision


def Recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    y_hat: pd.Series: Predicted values
    y: pd.Series: Actual values
    cls: Union[int, str]: Class to calculate recall for

    Returns: float: Recall
    """
    assert y_hat.size == y.size
    tp = ((y_hat == cls) & (y == cls)).sum()
    fn = ((y_hat != cls) & (y == cls)).sum()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return recall


def F1_Score(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    y_hat: pd.Series: Predicted values
    y: pd.Series: Actual values
    cls: Union[int, str]: Class to calculate F1 score for

    Returns: float: F1 score
    """
    assert y_hat.size == y.size
    prec = Precision(y_hat, y, cls)
    rec = Recall(y_hat, y, cls)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return f1


def MCC(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    y_hat: pd.Series: Predicted values
    y: pd.Series: Actual values
    cls: Union[int, str]: Class to calculate Matthews Correlation Coefficient for

    Returns: float: Matthews Correlation Coefficient
    """
    assert y_hat.size == y.size
    tp = (np.isclose(y_hat, cls, atol=0.01) & np.isclose(y, cls, atol=0.01)).sum()
    tn = (~np.isclose(y_hat, cls, atol=0.01) & ~np.isclose(y, cls, atol=0.01)).sum()
    fp = (np.isclose(y_hat, cls, atol=0.01) & ~np.isclose(y, cls, atol=0.01)).sum()
    fn = (~np.isclose(y_hat, cls, atol=0.01) & np.isclose(y, cls, atol=0.01)).sum()

    mcc_n = (tp * tn) - (fp * fn)
    mcc_d = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    mcc = mcc_n / mcc_d if mcc_d else 0
    return mcc


def RMSE(y_hat: pd.Series, y: pd.Series) -> float:
    """
    y_hat: pd.Series: Predicted values
    y: pd.Series: Actual values

    Returns: float: Root Mean Squared Error
    """
    assert y_hat.size == y.size
    rmse = np.sqrt(((y_hat - y) ** 2).mean())
    return rmse


def MAE(y_hat: pd.Series, y: pd.Series) -> float:
    """
    y_hat: pd.Series: Predicted values
    y: pd.Series: Actual values

    Returns: float: Mean Absolute Error
    """
    assert y_hat.size == y.size
    mae = abs(y_hat - y).mean()
    return mae