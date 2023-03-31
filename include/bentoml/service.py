import bentoml
import numpy as np

runner = bentoml.mlflow.get("iris-classifier:latest").to_runner()

svc = bentoml.Service('iris-classifier', runners=[runner])


@svc.api(input=bentoml.io.NumpyNdarray(), output=bentoml.io.NumpyNdarray())
def predict(input_val: np.ndarray):
    return runner.predict.run([input_val])[0]
