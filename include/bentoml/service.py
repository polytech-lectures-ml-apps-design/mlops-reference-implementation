import bentoml

runner = bentoml.mlflow.get("iris-classifier:latest").to_runner()

svc = bentoml.Service('iris-classifier', runners=[runner])


@svc.api(input=bentoml.io.NumpyNdarray(), output=bentoml.io.NumpyNdarray())
def predict(input_text: str):
    return runner.predict.run([input_text])[0]
