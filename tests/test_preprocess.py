import numpy as np
import pytest
from preprocessing import min_max
def test_check_values():
    arr=np.array([10,20,30])
    result=min_max(arr)
    assert result.min()==0
    assert result.max()==1

def test_check_negative():
    arr=np.array([-5,8,-2])
    result=min_max(arr)
    assert result.min()==0
    assert result.max()==1


def test_check_zero_size_arrray():
    with pytest.raises(ValueError):
        min_max(np.array([]))