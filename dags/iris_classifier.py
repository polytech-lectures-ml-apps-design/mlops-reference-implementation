from datetime import datetime, timedelta
from os.path import dirname, abspath
import os
import sys
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator, get_current_context

project_dir_path = dirname(dirname(abspath(__file__)))


def load_train_test_data(**kwargs):
    import os
    import dvc.api
    import pickle
    from sklearn.model_selection import train_test_split

    data_path = os.path.join(project_dir_path, "data/iris.pkl")

    data_url = dvc.api.get_url(path=data_path)

    with open(data_url, 'rb') as data_file:
        iris = pickle.load(data_file)

    X, y = iris.data, iris.target

    return train_test_split(X, y, test_size=0.2)


def _train_model():
    from sklearn import svm
    import mlflow
    import bentoml
    from mlflow.tracking import MlflowClient
    import git
    repo = git.Repo(project_dir_path, search_parent_directories=True)
    sha_commit = repo.head.object.hexsha

    x_train, x_test, y_train, y_test = load_train_test_data()

    model = svm.SVC(gamma='scale')

    with mlflow.start_run() as run:
        mlflow.set_tag('mlflow.source.git.commit', sha_commit)

        model.fit(x_train, y_train)
        model.score(x_test, y_test)

        mlflow.sklearn.log_model(model, "model")

        registered_model = mlflow.register_model(
            "runs:/{}/model".format(run.info.run_id), "IrisClassifier")

        bentoml.mlflow.import_model(
            "iris-classifier",
            registered_model.source,
            signatures={"predict": {"batchable": True}},
        )

    client = MlflowClient()

    client.transition_model_version_stage(
        name="IrisClassifier",
        version=registered_model.version,
        stage="Production",
    )

    context = get_current_context()

    context["ti"].xcom_push(key='model_uri', value=registered_model.source)

    return True


with DAG(
    'iris_classifier',
    description="Pipeline for training and deploying iris classifier",
    schedule=timedelta(days=1),
    start_date=datetime(2023, 3, 30),
    catchup=False,
    tags=["reference-implementation"]
) as dag:
    import os
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Iris classifier")
    mlflow.sklearn.autolog()

    # Start by versioning data and code to make sure results can be traced back to the data and code that generated it.
    # We assume the latest data has been loaded to under the data folder.
    # Our data does not change, but in a practical scenario, the data could have changed since last run.
    t1 = BashOperator(
        task_id="version_data",
        bash_command="cd {} && dvc add data && dvc push".format(
            project_dir_path)
    )

    # We version all code files using Git. The data folder won't be committed
    # because it was added to .gitignore by DVC
    t2 = BashOperator(
        task_id="stage_files",
        bash_command="cd {} && git add .".format(project_dir_path)
    )

    t3 = BashOperator(
        task_id="commit_files",
        bash_command="cd {} && git commit -m 'Update data' || echo 'No changes to commit'".format(
            project_dir_path)
    )

    t4 = ShortCircuitOperator(
        task_id="train_model",
        python_callable=_train_model
    )

    bento_path = os.path.join(project_dir_path, "include/bentoml")

    # We export BENTOML_MLFLOW_MODEL_PATH to be used in the bentofile.yaml
    # so BentoML can find the model's requirements.txt
    t5 = BashOperator(
        task_id="build_model",
        bash_command="cd {} && export BENTOML_MLFLOW_MODEL_PATH={{{{ ti.xcom_pull(key='model_uri') }}}} && echo $BENTOML_MLFLOW_MODEL_PATH && bentoml build".format(
            bento_path)
    )

    t6 = BashOperator(
        task_id="containerize_model",
        bash_command="bentoml containerize iris-classifier:latest -t iris-classifier:latest"
    )

    docker_compose_file_path = os.path.join(
        project_dir_path, "include/docker-compose/docker-compose.yml")

    t7 = BashOperator(
        task_id="serve_model",
        bash_command="docker compose -f {} up -d --wait".format(
            docker_compose_file_path)
    )

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7
