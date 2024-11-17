"""LLama workflows for Airflow."""

from datetime import timedelta
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

from llama_tuning.train import TrainerConfig

# Base configuration
DEFAULT_REGISTRY = "ghcr.io/unionai-oss"
IMAGE_TAG = "n0d4nltgwbb_iicwbsyfvq.."
BASE_IMAGE = f"{DEFAULT_REGISTRY}/flyte-llama-qlora:{IMAGE_TAG}"

# GCS paths
GCS_BUCKET = "airflow-llama-tuning-skypilot-375902"
DATASET_PATH = f"gs://{GCS_BUCKET}/llama/dataset"
MODEL_OUTPUT_PATH = f"gs://{GCS_BUCKET}/llama/models"

def setup_task(dag: DAG, task_id: str) -> KubernetesPodOperator:
    """Install required packages"""
    return KubernetesPodOperator(
        task_id=task_id,
        dag=dag,
        name=task_id,
        namespace='airflow',
        image=BASE_IMAGE,
        cmds=["python", "-c"],
        arguments=[
            """\
import subprocess
import sys
from pathlib import Path

packages = [
    "accelerate",
    "bitsandbytes",
    "datasets",
    "deepspeed==0.10.0",
    "gitpython",
    "huggingface_hub",
    "mwparserfromhell",
    "peft",
    "pandera",
    "pyyaml",
    "sentencepiece",
    "tokenizers",
    "transformers",
    "torch==2.0.1",
    "wandb",
    "google-cloud-storage"
]

subprocess.run(["pip", "install"] + packages, check=True)
dags_dir = Path("/opt/airflow/dags")
sys.path.append(str(dags_dir))
"""
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={
                'memory': '4Gi',
                'cpu': '1',
                'ephemeral-storage': '8Gi'
            }
        ),
        is_delete_operator_pod=True,
        get_logs=True,
    )

def create_dataset(dag: DAG, task_id: str, additional_urls: Optional[List[str]] = None) -> KubernetesPodOperator:
    """Create dataset task"""
    return KubernetesPodOperator(
        task_id=task_id,
        dag=dag,
        name=task_id,
        namespace='airflow',
        image=BASE_IMAGE,
        cmds=["python", "-c"],
        arguments=[
            f"""\
from pathlib import Path
import sys
sys.path.append("/opt/airflow/dags")

from llama_tuning.dataset import create_dataset, REPO_URLS
from google.cloud import storage

urls = {[*REPO_URLS, *(additional_urls or [])]}
working_dir = Path('/tmp/working_dir')
temp_output_dir = working_dir / "dataset"
repo_cache_dir = working_dir / "repo_cache"

# Create dataset locally first
create_dataset(urls, temp_output_dir, repo_cache_dir)

# Upload to GCS
client = storage.Client()
bucket = client.bucket("{GCS_BUCKET}")
for file_path in temp_output_dir.rglob("*"):
    if file_path.is_file():
        blob_path = f"llama/dataset/{file_path.relative_to(temp_output_dir)}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(file_path))

print("Dataset path: {DATASET_PATH}")
"""
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={
                'memory': '8Gi',
                'cpu': '2',
                'ephemeral-storage': '8Gi'
            }
        ),
        is_delete_operator_pod=True,
        get_logs=True,
    )

def train_model(
    dag: DAG,
    task_id: str,
    dataset_path: str,
    config: TrainerConfig,
    pretrained_adapter: Optional[Path] = None,
):
    """Train model task"""
    return KubernetesPodOperator(
        task_id=task_id,
        dag=dag,
        name=task_id,
        namespace='airflow',
        image=BASE_IMAGE,
        cmds=["python", "-c"],
        arguments=[
            f"""\
import os
import sys
from pathlib import Path
sys.path.append("/opt/airflow/dags")

from llama_tuning.train import train, TrainerConfig
from google.cloud import storage

os.environ["WANDB_API_KEY"] = '{Variable.get("WANDB_API_KEY")}'
os.environ["WANDB_PROJECT"] = "unionai-flyte-llama"
os.environ["WANDB_RUN_ID"] = "{task_id}"
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Download dataset from GCS
client = storage.Client()
bucket = client.bucket("{GCS_BUCKET}")
local_dataset_path = Path("/tmp/dataset")
for blob in bucket.list_blobs(prefix="llama/dataset/"):
    local_file = local_dataset_path / blob.name.replace("llama/dataset/", "")
    local_file.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_file))

config = TrainerConfig(**{config.__dict__})
config.data_dir = str(local_dataset_path)
config.output_dir = "{MODEL_OUTPUT_PATH}/{task_id}"

model = train(
    config,
    pretrained_adapter={pretrained_adapter},
    hf_auth_token='{Variable.get("HF_AUTH_TOKEN")}'
)

# Upload model to GCS
for file_path in Path(config.output_dir).rglob("*"):
    if file_path.is_file():
        blob_path = f"llama/models/{task_id}/{file_path.relative_to(config.output_dir)}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(file_path))
"""
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={
                'memory': '120Gi',
                'cpu': '44',
                'gpu': '8',
                'ephemeral-storage': '100Gi'
            }
        ),
        is_delete_operator_pod=True,
        get_logs=True,
    )

def publish_model(
    dag: DAG,
    task_id: str,
    model_dir: str,
    config: TrainerConfig,
):
    """Publish model task"""
    return KubernetesPodOperator(
        task_id=task_id,
        dag=dag,
        name=task_id,
        namespace='airflow',
        image=BASE_IMAGE,
        cmds=["python", "-c"],
        arguments=[
            f"""\
import sys
from pathlib import Path
sys.path.append("/opt/airflow/dags")

from llama_tuning.publish import publish_to_hf_hub
from llama_tuning.train import TrainerConfig
from google.cloud import storage

# Download model from GCS
client = storage.Client()
bucket = client.bucket("{GCS_BUCKET}")
local_model_dir = Path("/tmp/model")
for blob in bucket.list_blobs(prefix="{model_dir}"):
    local_file = local_model_dir / blob.name.replace("{model_dir}/", "")
    local_file.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_file))

config = TrainerConfig(**{config.__dict__})

publish_to_hf_hub(
    local_model_dir,
    config,
    hf_auth_token='{Variable.get("HF_AUTH_TOKEN")}'
)
"""
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={
                'memory': '10Gi',
                'cpu': '1',
                'ephemeral-storage': '64Gi'
            }
        ),
        is_delete_operator_pod=True,
        get_logs=True,
    )

def batch_size_tuning_workflow(
    dag: DAG,
    config: TrainerConfig,
    batch_sizes: List[int],
):
    """Batch size tuning workflow"""
    setup = setup_task(dag, "setup_dependencies")
    dataset_task = create_dataset(dag, "create_dataset")
    setup >> dataset_task # pylint: disable=pointless-statement
    
    train_tasks: List[KubernetesPodOperator] = []
    for batch_size in batch_sizes:
        config_copy = replace(config, batch_size=batch_size)
        train_task = train_model(
            dag=dag,
            task_id=f"train_batch_size_{batch_size}",
            dataset_path=DATASET_PATH,
            config=config_copy,
        )
        train_tasks.append(train_task)
        dataset_task >> train_task # pylint: disable=pointless-statement

    return train_tasks

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'llama_fine_tuning',
    default_args=default_args,
    description='LLama fine tuning workflow',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['llama', 'fine-tuning'],
) as dag:
    
    # Create base config
    config = TrainerConfig(
        model_path='codellama/CodeLlama-7b-hf',
        data_dir=DATASET_PATH,
        output_dir=MODEL_OUTPUT_PATH,
        num_epochs=20,
        batch_size=8,
        model_max_length=1024,
        use_4bit=True,
        use_qlora=True,
        report_to="wandb",
    )

    # Define batch sizes to try
    batch_sizes = [4, 8, 16]

    # Create workflow
    train_tasks = batch_size_tuning_workflow(
        dag=dag,
        config=config,
        batch_sizes=batch_sizes,
    )

    # Optional: Add publish task for best model
    publish = publish_model(
        dag=dag,
        task_id="publish_model",
        model_dir=f"{MODEL_OUTPUT_PATH}/train_batch_size_8",  # Assuming batch_size 8 is best
        config=config,
    )

    # Set dependencies
    train_tasks[1] >> publish  # Connect batch_size 8 task to publish