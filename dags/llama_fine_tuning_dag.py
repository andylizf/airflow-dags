"""
LLaMA fine-tuning DAG for Airflow running on GCP GKE.
"""

from datetime import datetime, timedelta
from pathlib import Path
import json
import os

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from kubernetes.client import models as k8s

# 基础配置
DEFAULT_REGISTRY = "ghcr.io/unionai-oss"
IMAGE_TAG = "n0d4nltgwbb_iicwbsyfvq.."
BASE_IMAGE = f"{DEFAULT_REGISTRY}/flyte-llama-qlora:{IMAGE_TAG}"

# GCS路径配置
GCS_BUCKET = "airflow-llama-tuning-skypilot-375902"
DATASET_PATH = f"gs://{GCS_BUCKET}/llama/dataset"
MODEL_OUTPUT_PATH = f"gs://{GCS_BUCKET}/llama/models"

# 资源配置
GPU_LIMITS = {
    "nvidia.com/gpu": "8"  # 根据需要调整GPU数量
}

# Secret配置
WANDB_API_KEY = "wandb-api-key"  # 从Airflow Variables获取
HF_AUTH_TOKEN = "hf-auth-token"  # 从Airflow Variables获取

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# 创建volume mounts和volumes配置
volume_mounts = [
    k8s.V1VolumeMount(
        name="gcs-fuse-volume",
        mount_path="/gcs",
        read_only=False
    ),
    k8s.V1VolumeMount(
        name="dshm",
        mount_path="/dev/shm"
    )
]

volumes = [
    k8s.V1Volume(
        name="gcs-fuse-volume",
        persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
            claim_name="gcs-fuse-pvc"
        )
    ),
    k8s.V1Volume(
        name="dshm",
        empty_dir=k8s.V1EmptyDirVolumeSource(
            medium="Memory"
        )
    )
]

with DAG(
    'llama_fine_tuning',
    default_args=default_args,
    description='LLaMA Fine-tuning Pipeline',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['llm', 'fine-tuning'],
) as dag:

    # 创建数据集任务
    create_dataset = KubernetesPodOperator(
        task_id='create_dataset',
        name='create-dataset',
        namespace='airflow',
        image=BASE_IMAGE,
        cmds=["python", "-c"],
        arguments=[
            """
            import flyte_llama.dataset as dataset
            from pathlib import Path
            
            output_dir = Path('/gcs/dataset')
            repo_cache_dir = Path('/tmp/repo_cache')
            
            dataset.create_dataset(
                dataset.REPO_URLS,
                output_dir,
                repo_cache_dir
            )
            """
        ],
        container_resources={
            'memory': '8Gi',
            'cpu': '2',
            'ephemeral-storage': '8Gi'
        },
        volume_mounts=volume_mounts,
        volumes=volumes,
        is_delete_operator_pod=True,
        get_logs=True,
    )

    # 训练任务
    train_model = KubernetesPodOperator(
        task_id='train_model',
        name='train-model',
        namespace='airflow',
        image=BASE_IMAGE,
        cmds=["python", "-c"],
        arguments=[
            f"""
            import os
            import json
            from pathlib import Path
            import flyte_llama.train as train
            
            # 设置环境变量
            os.environ['WANDB_API_KEY'] = '{Variable.get(WANDB_API_KEY)}'
            os.environ['WANDB_PROJECT'] = 'llama-fine-tuning'
            os.environ['TRANSFORMERS_CACHE'] = '/tmp'
            os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
            os.environ['TOKENIZERS_PARALLELISM'] = 'true'
            
            # 加载配置
            config = train.TrainerConfig(
                model_path='codellama/CodeLlama-7b-hf',
                data_dir='/gcs/dataset',
                output_dir='/gcs/models',
                num_epochs=20,
                batch_size=8,
                model_max_length=1024,
                use_4bit=True,
                use_qlora=True,
            )
            
            # 开始训练
            train.train(
                config,
                pretrained_adapter=None,
                hf_auth_token='{Variable.get(HF_AUTH_TOKEN)}'
            )
            """
        ],
        container_resources={
            'memory': '120Gi',
            'cpu': '44',
            'ephemeral-storage': '100Gi',
            'nvidia.com/gpu': '8'
        },
        volume_mounts=volume_mounts,
        volumes=volumes,
        is_delete_operator_pod=True,
        get_logs=True,
    )

    # 发布模型任务
    publish_model = KubernetesPodOperator(
        task_id='publish_model',
        name='publish-model',
        namespace='airflow',
        image=BASE_IMAGE,
        cmds=["python", "-c"],
        arguments=[
            f"""
            from pathlib import Path
            import flyte_llama.publish as publish
            import flyte_llama.train as train
            
            model_dir = Path('/gcs/models')
            config = train.TrainerConfig(
                model_path='codellama/CodeLlama-7b-hf',
                output_dir='/gcs/models',
                publish_config=train.PublishConfig(
                    repo_id='your-hf-repo-id'  # 替换为你的HuggingFace repo ID
                )
            )
            
            publish.publish_to_hf_hub(
                model_dir,
                config,
                hf_auth_token='{Variable.get(HF_AUTH_TOKEN)}'
            )
            """
        ],
        container_resources={
            'memory': '10Gi',
            'cpu': '1',
            'ephemeral-storage': '64Gi'
        },
        volume_mounts=volume_mounts,
        volumes=volumes,
        is_delete_operator_pod=True,
        get_logs=True,
    )

    # 设置任务依赖
    create_dataset >> train_model >> publish_model 