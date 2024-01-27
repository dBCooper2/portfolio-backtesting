import pandas as pd

class OLS_Regression:
    def __init__(self, df: pd.DataFrame, iterations: int, learning_rate: float) -> None:
        self.df = df
        self.i = iterations
        self.lr = learning_rate
        self.m = 0
        self.b = 0

    def gradient_descent(self, m_current, b_current):
        m_gradient = 0
        b_gradient = 0

        n = len(self.df) # The number of rows in the dataset

        # Calculate the partial derivative summations
        for i in range(n):
            x = self.df.iloc[i].x
            y = self.df.iloc[i].y

            # These are a pythonic representation of partial derivative equations found in the theory section
            m_gradient += (-2/n) * x * (y - (m_current * x + b_current))
            b_gradient += (-2/n) * (y - (m_current * x + b_current))

        # Calculate the Gradient Descent equations from the theory section
        self.m = m_current - self.learning_rate * m_gradient
        self.b = b_current - self.learning_rate * b_gradient

        return self.m,self.b
    
    def run_regression(self):
        self.m = 0
        self.b = 0

        for i in range(self.i):
            self.m,self.b = self.gradient_descent(self.m, self.b)

        return self.m,self.b