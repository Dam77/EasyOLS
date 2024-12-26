from backend.inputs import inputs
from backend.estimations import Final_answer

X, Y, t = inputs()

if __name__ == '__main__':
    Final_answer(X,Y)