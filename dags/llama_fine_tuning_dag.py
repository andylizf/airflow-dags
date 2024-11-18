"""LLama workflows for Airflow."""

from datetime import timedelta, datetime
from pathlib import Path
from typing import List, Optional, Any

from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

# Base paths
DAGS_DIR = Path(__file__).parent
LLAMA_TUNING_DIR = DAGS_DIR / 'llama_tuning'

def read_module_code(module_path: Path) -> str:
    """Read module code and handle imports."""
    with open(module_path) as f:
        lines = f.readlines()
    
    # Filter out the if __main__ block and collect imports
    filtered_lines: List[str] = []
    imports: List[str] = []
    in_main_block = False
    
    for line in lines:
        if line.strip().startswith('from llama_tuning.'):
            # Convert internal imports to direct imports
            module_name = line.split('from llama_tuning.')[1].split(' import ')[0]
            imports.append(module_name)
            continue
        if line.strip().startswith('if __name__ == "__main__"'):
            in_main_block = True
            continue
        if not in_main_block:
            filtered_lines.append(line)
    
    # Read and inject imported modules
    imported_code: list[str] = []
    for module_name in imports:
        module_path = LLAMA_TUNING_DIR / f'{module_name}.py'
        with open(module_path) as f:
            module_code = f.read()
            # Remove any llama_tuning imports from imported modules
            module_code = '\n'.join(
                line for line in module_code.split('\n')
                if not line.strip().startswith('from llama_tuning.')
            )
            imported_code.append(f"\n# Imported from {module_name}.py\n{module_code}")
    
    return ''.join(imported_code + filtered_lines)

# Base configuration
DEFAULT_REGISTRY = "ghcr.io/unionai-oss"
IMAGE_TAG = "n0d4nltgwbb_iicwbsyfvq.."
BASE_IMAGE = f"{DEFAULT_REGISTRY}/flyte-llama-qlora:{IMAGE_TAG}"

# GCS paths
GCS_BUCKET = "airflow-llama-tuning-skypilot-375902"
DATASET_PATH = f"gs://{GCS_BUCKET}/llama/dataset"
MODEL_OUTPUT_PATH = f"gs://{GCS_BUCKET}/llama/models"

# Load module code
TRAIN_CODE = read_module_code(LLAMA_TUNING_DIR / 'train.py')
DATASET_CODE = read_module_code(LLAMA_TUNING_DIR / 'dataset.py')
PUBLISH_CODE = read_module_code(LLAMA_TUNING_DIR / 'publish.py')

# GCP Volume Configuration
GCP_VOLUME_CONFIG = {
    "env_vars": {
        "GOOGLE_APPLICATION_CREDENTIALS": "/var/secrets/google/key.json"
    },
    "volumes": [
        k8s.V1Volume(
            name="gcp-key",
            secret=k8s.V1SecretVolumeSource(
                secret_name="gcp-key"
            )
        )
    ],
    "volume_mounts": [
        k8s.V1VolumeMount(
            name="gcp-key",
            mount_path="/var/secrets/google",
            read_only=True
        )
    ]
}

def create_dataset(dag: DAG, task_id: str, additional_urls: Optional[List[str]] = None,
                   force_create: bool = False) -> KubernetesPodOperator:
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
import hashlib
import json
from pathlib import Path
from typing import List, Optional
from google.cloud import storage

# Inject dataset module code
{DATASET_CODE}

# Calculate config hash
urls = REPO_URLS + {additional_urls or []}
config_hash = hashlib.sha256(
    json.dumps({{"urls": sorted(urls)}}, sort_keys=True).encode()
).hexdigest()[:8]

# Check if dataset with same hash exists
client = storage.Client()
bucket = client.bucket("{GCS_BUCKET}")
hash_blob = bucket.blob("llama/dataset/.hash")
dataset_exists = False

if hash_blob.exists():
    existing_hash = hash_blob.download_as_text().strip()
    dataset_exists = existing_hash == config_hash

if dataset_exists and not {force_create}:
    print(f"Dataset with hash {{config_hash}} already exists, skipping creation...")
else:
    print(f"Creating new dataset with hash {{config_hash}}...")
    working_dir = Path('/tmp/working_dir')
    temp_output_dir = working_dir / "dataset"
    repo_cache_dir = working_dir / "repo_cache"

    # Create dataset locally first
    create_dataset(urls, temp_output_dir, repo_cache_dir)

    # Upload to GCS
    for file_path in temp_output_dir.rglob("*"):
        if file_path.is_file():
            blob_path = f"llama/dataset/{{file_path.relative_to(temp_output_dir)}}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(file_path))
    
    # Save hash
    hash_blob.upload_from_string(config_hash)

print("Dataset path: {DATASET_PATH}")"""
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={
                'memory': '8Gi',
                'cpu': '2',
                'ephemeral-storage': '8Gi'
            },
            limits={
                'memory': '8Gi',
                'cpu': '2',
                'ephemeral-storage': '8Gi'
            }
        ),
        is_delete_operator_pod=True,
        get_logs=True,
        **GCP_VOLUME_CONFIG
    )

def train_model(
    dag: DAG,
    task_id: str,
    dataset_path: str,
    config: dict[str, Any],
    pretrained_adapter: Optional[Path] = None,
) -> KubernetesPodOperator:
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
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass
from google.cloud import storage

# Inject train module code
{TRAIN_CODE}

os.environ["WANDB_API_KEY"] = '{Variable.get("wandb-api-key")}'
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

config = TrainerConfig(**{config})
config.data_dir = str(local_dataset_path)
config.output_dir = "{MODEL_OUTPUT_PATH}/{task_id}"

model = train(
    config,
    pretrained_adapter={pretrained_adapter},
    hf_auth_token='{Variable.get("hf-auth-token")}'
)

# Upload model to GCS
for file_path in Path(config.output_dir).rglob("*"):
    if file_path.is_file():
        blob_path = f"llama/models/{{task_id}}/{{file_path.relative_to(config.output_dir)}}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(file_path))"""
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={
                'cpu': '4',
                'memory': '16Gi',
                'ephemeral-storage': '20Gi',
                'nvidia.com/gpu': '1'
            },
            limits={
                'cpu': '6',
                'memory': '24Gi',
                'ephemeral-storage': '20Gi',
                'nvidia.com/gpu': '1'
            }
        ),
        is_delete_operator_pod=True,
        get_logs=True,
        node_selector={
            'cloud.google.com/gke-nodepool': 'gpu-pool'
        },
        tolerations=[
            k8s.V1Toleration(
                key='nvidia.com/gpu',
                operator='Exists',
                effect='NoSchedule'
            )
        ],
        startup_timeout_seconds=300,
        **GCP_VOLUME_CONFIG
    )

def publish_model(
    dag: DAG,
    task_id: str,
    model_dir: str,
    config: dict[str, Any],
) -> KubernetesPodOperator:
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

config = TrainerConfig(**{config})

publish_to_hf_hub(
    local_model_dir,
    config,
    hf_auth_token='{Variable.get("hf-auth-token")}'
)
"""
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={
                'memory': '10Gi',
                'cpu': '1',
                'ephemeral-storage': '64Gi'
            },
            limits={
                'memory': '10Gi',
                'cpu': '1',
                'ephemeral-storage': '64Gi'
            }
        ),
        is_delete_operator_pod=True,
        get_logs=True,
        **GCP_VOLUME_CONFIG
    )

def batch_size_tuning_workflow(
    dag: DAG,
    config: dict[str, Any],
    batch_sizes: List[int],
):
    """Batch size tuning workflow"""
    dataset_task = create_dataset(dag, "create_dataset")
    
    train_tasks: List[KubernetesPodOperator] = []
    for batch_size in batch_sizes:
        # Create a copy of the config dictionary and update the batch_size
        config_copy = config.copy()
        config_copy['batch_size'] = batch_size
        
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
    config = {
        'model_path': 'codellama/CodeLlama-7b-hf',
        'data_dir': DATASET_PATH,
        'output_dir': MODEL_OUTPUT_PATH,
        'num_epochs': 20,
        'batch_size': 8,
        'model_max_length': 1024,
        'use_4bit': True,
        'use_qlora': True,
        'report_to': "wandb",
    }

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