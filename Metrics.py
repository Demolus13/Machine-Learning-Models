'''
Metrics For Machine Learning Models
'''

from typing import Union
import torch

def Accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    y_hat: torch.Tensor: Predicted values
    y: torch.Tensor: Actual values

    Returns: float: Accuracy
    """
    
    assert y_hat.shape == y.shape
    accuracy = (y_hat == y).float().mean().item()
    return accuracy


def Precision(y_hat: torch.Tensor, y: torch.Tensor, cls: Union[int, str], tol: float = 1e-6) -> float:
    """
    y_hat: torch.Tensor: Predicted values
    y: torch.Tensor: Actual values
    cls: Union[int, str]: Class to calculate precision for
    tol: float: Tolerance for comparison

    Returns: float: Precision
    """

    assert y_hat.shape == y.shape
    tp = (torch.isclose(y_hat, cls, atol=tol) & torch.isclose(y, cls, atol=tol)).sum().item()
    fp = (torch.isclose(y_hat, cls, atol=tol) & ~torch.isclose(y, cls, atol=tol)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return precision


def Recall(y_hat: torch.Tensor, y: torch.Tensor, cls: Union[int, str], tol: float = 1e-6) -> float:
    """
    y_hat: torch.Tensor: Predicted values
    y: torch.Tensor: Actual values
    cls: Union[int, str]: Class to calculate recall for
    tol: float: Tolerance for comparison

    Returns: float: Recall
    """

    assert y_hat.shape == y.shape
    tp = (torch.isclose(y_hat, cls, atol=tol) & torch.isclose(y, cls, atol=tol)).sum().item()
    fn = (~torch.isclose(y_hat, cls, atol=tol) & torch.isclose(y, cls, atol=tol)).sum().item()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return recall


def F1_Score(y_hat: torch.Tensor, y: torch.Tensor, cls: Union[int, str]) -> float:
    """
    y_hat: torch.Tensor: Predicted values
    y: torch.Tensor: Actual values
    cls: Union[int, str]: Class to calculate F1 score for

    Returns: float: F1 score
    """

    assert y_hat.shape == y.shape
    prec = Precision(y_hat, y, cls)
    rec = Recall(y_hat, y, cls)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return f1


def MCC(y_hat: torch.Tensor, y: torch.Tensor, cls: Union[int, str], tol: float = 1e-6) -> float:
    """
    y_hat: torch.Tensor: Predicted values
    y: torch.Tensor: Actual values
    cls: Union[int, str]: Class to calculate Matthews Correlation Coefficient for
    tol: float: Tolerance for comparison

    Returns: float: Matthews Correlation Coefficient
    """

    assert y_hat.shape == y.shape
    tp = (torch.isclose(y_hat, cls, atol=tol) & torch.isclose(y, cls, atol=tol)).sum().float()
    tn = (~torch.isclose(y_hat, cls, atol=tol) & ~torch.isclose(y, cls, atol=tol)).sum().float()
    fp = (torch.isclose(y_hat, cls, atol=tol) & ~torch.isclose(y, cls, atol=tol)).sum().float()
    fn = (~torch.isclose(y_hat, cls, atol=tol) & torch.isclose(y, cls, atol=tol)).sum().float()

    mcc_n = (tp * tn) - (fp * fn)
    mcc_d = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    mcc = mcc_n / mcc_d if mcc_d > 0 else 0
    return mcc.item()


def RMSE(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    y_hat: torch.Tensor: Predicted values
    y: torch.Tensor: Actual values

    Returns: float: Root Mean Squared Error
    """

    assert y_hat.shape == y.shape
    rmse = torch.sqrt(((y_hat - y) ** 2).mean()).item()
    return rmse


def MAE(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    y_hat: torch.Tensor: Predicted values
    y: torch.Tensor: Actual values

    Returns: float: Mean Absolute Error
    """

    assert y_hat.shape == y.shape
    mae = torch.abs(y_hat - y).mean().item()
    return mae