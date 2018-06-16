import os
import numpy as np

from sklearn.externals import joblib


def maybe_fit(model, X, y, model_dump):
    if not os.path.exists(model_dump):
        model.fit(X, y)
        joblib.dump(model, model_dump)
    else:
        model = joblib.load(model_dump)
    return model


def maybe_predict(model, X, result_dump):
    if not os.path.exists(result_dump):
        result = model.predict(X)
        np.save(result_dump, result)
    else:
        result = np.load(result_dump)
    return result