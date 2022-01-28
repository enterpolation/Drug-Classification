import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def prepare_data(df):
    df['Drug'].replace({'DrugY': 'drugY'}, inplace=True)
    df = df[df['Na_to_K'] <= 32]
    target = df['Drug']
    sample = pd.get_dummies(df.drop(['Drug', 'Sex'], axis=1))
    return np.array(sample), np.array(target)


if __name__ == "__main__":
    data = pd.read_csv('drug200.csv')

    X, y = prepare_data(data)

    model = LogisticRegression(
        max_iter=10000,
        penalty='l1',
        solver='liblinear',
        C=10
    )

    model.fit(X, y)

    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f)
