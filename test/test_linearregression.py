import numpy as np
from luvex import Linearregression

def test_linear_regression():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    model = Linearregression()
    model.fit(X,y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y)

if __name__ == "__main__":
    test_linear_regression()
    print("All tests passed for My_Linear_Regression")