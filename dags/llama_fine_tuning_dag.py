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

# 读取原始代码文件
def read_original_code(file_name: str) -> str:
    """Read original source code without processing imports."""
    with open(LLAMA_TUNING_DIR / file_name) as f:
        return f.read()

# 初始化原始代码常量
ORIGINAL_DATALOADER_CODE = read_original_code('dataloader.py')
ORIGINAL_TRAIN_CODE = read_original_code('train.py').replace(
    'from llama_tuning.dataloader import',
    'from .dataloader import'
)
ORIGINAL_DATASET_CODE = read_original_code('dataset.py')
ORIGINAL_PUBLISH_CODE = read_original_code('publish.py')
ORIGINAL_INFERENCE_CODE = read_original_code('inference.py')
ORIGINAL_FETCHER_CODE = read_original_code('fetcher.py')


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
DEFAULT_REGISTRY = "andylizf"
BASE_IMAGE = f"{DEFAULT_REGISTRY}/flyte-llama-qlora"

# GCS paths
GCS_BUCKET = "airflow-llama-tuning-skypilot-375902"
DATASET_PATH = f"gs://{GCS_BUCKET}/llama/dataset"
MODEL_OUTPUT_PATH   = f"gs://{GCS_BUCKET}/llama/models"

# Load module code
TRAIN_CODE = read_module_code(LLAMA_TUNING_DIR / 'train.py')
DATASET_CODE = read_module_code(LLAMA_TUNING_DIR / 'dataset.py')
PUBLISH_CODE = read_module_code(LLAMA_TUNING_DIR / 'publish.py')

INFERENCE_CODE = read_module_code(LLAMA_TUNING_DIR / 'inference.py')
FETCHER_CODE = read_module_code(LLAMA_TUNING_DIR / 'fetcher.py')

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
        startup_timeout_seconds=600,
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
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass
from google.cloud import storage
import shlex
import subprocess
import os

# 创建模块目录
module_dir = Path('/tmp/llama_train')
module_dir.mkdir(parents=True, exist_ok=True)
(module_dir / "__init__.py").touch()

# 写入训练相关代码
train_code = r'''{ORIGINAL_TRAIN_CODE}'''
(module_dir / "train.py").write_text(train_code)

dataloader_code = r'''{ORIGINAL_DATALOADER_CODE}'''
(module_dir / "dataloader.py").write_text(dataloader_code)

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


# 使用 torchrun 启动训练，使用 -m 参数
cmd = [
    "torchrun",
    "--nproc_per_node=8",
    "--master_port=29500",
    "-m", "llama_train.train",  # 使用模块路径
    "--config", json.dumps(asdict(config)),
    "--hf_auth_token", '{Variable.get("hf-auth-token")}'
]

print(f"Starting training with command: {{' '.join(cmd)}}")
try:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env={{
            **os.environ,
            "PYTHONUNBUFFERED": "1"
        }},
        cwd="/tmp"  # 设置工作目录
    )

    # 实时输出日志
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        print(line, end='', flush=True)

    # 等待进程完成
    process.wait()
    if process.returncode != 0:
        time.sleep(3600)
        raise Exception(f"Training failed with return code {{process.returncode}}")
except Exception as e:
    print(f"Error during training: {{str(e)}}")
    raise

# Upload model to GCS
for file_path in Path(config.output_dir).rglob("*"):
    if file_path.is_file():
        blob_path = f"llama/models/{task_id}/{{file_path.relative_to(config.output_dir)}}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(file_path))
"""
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={
                'cpu': '12',
                'memory': '64Gi',
                'ephemeral-storage': '40Gi',
                'nvidia.com/gpu': '8'  # 使用全部 8 个 GPU
            },
            limits={
                'cpu': '16',          # 给一些 CPU 弹性
                'memory': '128Gi',     # 给一些内存弹性
                'ephemeral-storage': '40Gi',
                'nvidia.com/gpu': '8'
            }),
        is_delete_operator_pod=False,
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
        startup_timeout_seconds=600,
        volumes=[
            # 共享内存
            k8s.V1Volume(
                name='dshm',
                empty_dir=k8s.V1EmptyDirVolumeSource(
                    medium='Memory',
                    size_limit='32Gi'
                )
            ),
            **GCP_VOLUME_CONFIG.get('volumes', {})
        ],
        volume_mounts=[
            k8s.V1VolumeMount(
                name='dshm',
                mount_path='/dev/shm'
            ),
            **GCP_VOLUME_CONFIG.get('volume_mounts', {})
        ],
        env_vars={
            # GKE的默认NVIDIA库路径
            'LD_LIBRARY_PATH': '/usr/local/nvidia/lib64:/usr/local/cuda/lib64',
            'NVIDIA_VISIBLE_DEVICES': 'all',
            'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility',
            'PYTHONUNBUFFERED': '1',
            "GOOGLE_APPLICATION_CREDENTIALS": "/var/secrets/google/key.json"
            **os.environ
        },
        # 使用NVIDIA运行时
        security_context=k8s.V1PodSecurityContext(
            run_as_user=0,
            fs_group=0
        ),
    )

def filter_issues(
    dag: DAG,
    task_id: str,
    dataset_path: str,
    model_path: str,
) -> KubernetesPodOperator:
    """Filter issues task"""
    return KubernetesPodOperator(
        task_id=task_id,
        dag=dag,
        name=task_id,
        namespace='airflow',
        image=BASE_IMAGE,
        cmds=["python", "-c"],
        arguments=[
            f"""\
import json
import os
from pathlib import Path
from google.cloud import storage
from typing import List, Dict
from datetime import datetime

# Download dataset from GCS
client = storage.Client()
bucket = client.bucket("{GCS_BUCKET}")
local_dataset_path = Path("/tmp/dataset")
for blob in bucket.list_blobs(prefix="{dataset_path}"):
    local_file = local_dataset_path / blob.name.replace("{dataset_path}/", "")
    local_file.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_file))

# Set up output directory
filtered_dir = Path("/tmp/filtered_issues")
filtered_dir.mkdir(parents=True, exist_ok=True)

os.environ["HUGGING_FACE_HUB_TOKEN"] = '{Variable.get("hf-auth-token")}'

# Load issues and discussions
issues_dir = local_dataset_path / "issues"
discussions_dir = local_dataset_path / "discussions"

issues = []
discussions = []

from typing import List
from dataclasses import dataclass

@dataclass
class Issue:
    title: str
    body: str
    number: int
    state: str
    comments: List[str]
    url: str

for issue_file in issues_dir.glob("*.json"):
    with issue_file.open() as f:
        data = json.load(f)
        issues.append(Issue(**data))

for disc_file in discussions_dir.glob("*.json"):
    with disc_file.open() as f:
        discussions.append(json.load(f))

# Filter issues using LLM
{FETCHER_CODE}

filtered_items, classification_results = filter_issues_with_llm(
    issues,
    discussions,
    "{model_path}",
    token='{Variable.get("hf-auth-token")}'
)

# Save classification results
with (filtered_dir / "classification_results.json").open("w") as f:
    json.dump({{
        "total_items": len(issues) + len(discussions),
        "filtered_items": len(filtered_items),
        "results": classification_results,
        "timestamp": str(datetime.now().isoformat())
    }}, f, indent=2)

# Save filtered issues
saved_files = []
for item in filtered_items:
    if isinstance(item, Issue):
        output_file = filtered_dir / f"filtered_issue_{{item.number}}.json"
    else:  # Discussion
        output_file = filtered_dir / f"filtered_discussion_{{item.number}}.json"
    
    with output_file.open("w") as f:
        json.dump(item.__dict__, f, indent=2) 
    saved_files.append(output_file)

# Save manifest
with (filtered_dir / "manifest.json").open("w") as f:
    json.dump({{
        "total_issues": len(issues),
        "filtered_items": len(filtered_items),
        "files": [f.name for f in saved_files],
        "timestamp": str(datetime.now().isoformat()),
        "model_path": "{model_path}",
        "classification_summary": {{
            "accepted": len(filtered_items),
            "rejected": len(issues) - len(filtered_items)
        }}
    }}, f, indent=2)

# Upload results to GCS
for file_path in filtered_dir.rglob("*"):
    if file_path.is_file():
        blob_path = f"llama/filtered_issues/{{file_path.relative_to(filtered_dir)}}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(file_path))
"""
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={
                'memory': '8Gi',
                'cpu': '2',
                'ephemeral-storage': '20Gi',
                'nvidia.com/gpu': '1'
            },
            limits={
                'memory': '8Gi',
                'cpu': '2',
                'ephemeral-storage': '20Gi',
                'nvidia.com/gpu': '1'
            }
        ),
        # is_delete_operator_pod=False,
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
        startup_timeout_seconds=600,
        **GCP_VOLUME_CONFIG
    )

def generate_answers(
    dag: DAG,
    task_id: str,
    filtered_issues_path: str,
    base_model_path: str,
    adapter_path: str,
) -> KubernetesPodOperator:
    """Generate answers task"""

    system_prompt = """You are an expert programmer analyzing a feature request for the SkyPilot framework. Break down this feature request into a clear implementation plan.

IMPORTANT:
1. Do not include commit hashes, version numbers, or any placeholder text like "Commit hash".
2. Be concise and to the point, do not include any other information.
3. Use markdown syntax for code blocks and lists.
4. Do not write like an email, just write the plan.

Here's an example analysis of a fake feature request:
----------------
Feature Request:
Title: Add timeout parameter for spot job recovery
Description: When using spot instances, if the instance is preempted, SkyPilot will try to recover indefinitely. We should add a timeout parameter to limit the recovery time.

Analysis:
1. Current Behavior:
- Spot recovery loop runs without timeout control:
```python
# spot.py
def recover_spot_job(task: Task):
    while True:  # Infinite loop without timeout
        try:
            return _attempt_spot_recovery(task)
        except Exception as e:
            logger.error(f'Recovery failed: {e}')
            time.sleep(RETRY_INTERVAL)
```
- Task class has no timeout configuration:
```python
class Task:
    def __init__(self, spot_policy: Optional[str] = None):
        self.spot_policy = spot_policy  # No timeout parameter
```
- Users can only interrupt recovery manually by killing the process

2. Proposed Solution:
- Add spot_recovery_timeout parameter to job submission
- Implement timeout logic in spot recovery loop
- Provide clear error message when timeout is reached

3. Implementation Plan:
- Modify spot.py to add timeout parameter and logic
- Update task.py to include timeout configuration
- Add timeout handling in recovery loop

4. Code Implementation:
- Add timeout parameter to Task class:
```python
class Task:
    def __init__(self, spot_recovery_timeout: Optional[int] = None):
        self.spot_recovery_timeout = spot_recovery_timeout
```

- Implement timeout logic in spot recovery:
```python
def recover_spot_job(task: Task):
    start_time = time.time()
    while True:
        if (task.spot_recovery_timeout and 
            time.time() - start_time > task.spot_recovery_timeout):
            raise SpotTimeoutError(
                f'Spot recovery timeout after {task.spot_recovery_timeout}s')
        try:
            return _attempt_spot_recovery(task)
        except Exception as e:
            logger.error(f'Recovery failed: {e}')
```

- Add unit tests:
```python
def test_spot_recovery_timeout():
    task = Task(spot_recovery_timeout=60)
    with pytest.raises(SpotTimeoutError):
        recover_spot_job(task)
```

----------------

Provide a similar detailed analysis with concrete code examples.

1. Current Behavior:
- Describe current implementation with relevant code snippets in SkyPilot repository
- Identify limitations in existing code
- Show where changes will be needed

2. Proposed Solution:
- Core functionality needed
- Key benefits
- Technical requirements

3. Implementation Plan:
- Components to modify
- Key steps (max 3)
- Integration points

4. Code Implementation:
- Key function/class changes with code examples
- Integration code
- Testing approach with example test cases

Beep each section brief and focused. Prioritize practical implementation details."""

    return KubernetesPodOperator(
        task_id=task_id,
        dag=dag,
        name=task_id,
        namespace='airflow',
        image=BASE_IMAGE,
        cmds=["python", "-c"],
        arguments=[
            f"""\
import json
import os
from pathlib import Path
from google.cloud import storage
import transformers
from datetime import datetime
print(f"Transformers version: {{transformers.__version__}}")

# Set up environment
os.environ["HUGGING_FACE_HUB_TOKEN"] = '{Variable.get("hf-auth-token")}'

# Download filtered issues from GCS
client = storage.Client()
bucket = client.bucket("{GCS_BUCKET}")
issues_path = Path("/tmp/filtered_issues")
issues_path.mkdir(parents=True, exist_ok=True)

for blob in bucket.list_blobs(prefix="{filtered_issues_path}"):
    local_file = issues_path / blob.name.replace("{filtered_issues_path}/", "")
    local_file.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_file))

# Load issues
issues = []
for issue_file in issues_path.glob("filtered_issue_*.json"):
    with issue_file.open() as f:
        try:
            issue_data = json.load(f)
            issues.append(issue_data)
        except Exception as e:
            print(f"Error loading {{issue_file}}: {{e}}")

if len(issues) == 0:
    issues = [{{"number": 0, "url": "https://github.com/skypilot-org/skypilot/issues/test", "title": "How to launch a cluster in SkyPilot", "body": "I want to know how to launch a cluster using SkyPilot. Please provide detailed steps and code examples."}}]

print(f"Loaded {{len(issues)}} issues")

# Download adapter if needed
adapter_local_path = Path("/tmp/adapter")
adapter_local_path.mkdir(parents=True, exist_ok=True)
for blob in bucket.list_blobs(prefix="{adapter_path}"):
    local_file = adapter_local_path / blob.name.replace("{adapter_path}/", "")
    local_file.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_file))

# 在下载 adapter 文件后
print("\\nAdapter files downloaded:")
for f in adapter_local_path.glob("*"):
    print(f"  {{f}}")
    if f.is_file():
        print(f"  File size: {{f.stat().st_size}} bytes")

# Configure model
{INFERENCE_CODE}

serving_config = ServingConfig(
    model_path="{base_model_path}",
    adapter_path=str(adapter_local_path),
    device_map="auto",
    hf_token='{Variable.get("hf-auth-token")}',
)

# 在加载模型前
print("\\nStarting model loading...")
print(f"Base model path: {base_model_path}")
print(f"Adapter path: {{adapter_local_path}}")
print(f"Device map: {{serving_config.device_map}}")

# Load model and tokenizer
model, tokenizer = load_pipeline(serving_config)

# 在加载模型后
print("\\nModel loaded successfully")
print(f"Model config: {{model.config}}")

# # Add test question
# test_issue = {{
#     "number": 0,
#     "url": "https://github.com/skypilot-org/skypilot/issues/test",
#     "title": "How to launch a cluster in SkyPilot",
#     "body": "I want to know how to launch a cluster using SkyPilot. Please provide detailed steps and code examples."
# }}
# issues.insert(0, test_issue)

system_prompt = '''{system_prompt}'''

answers = []
for issue in issues[:100]:
    try:
        issue_prompt = f"Title: {{issue['title']}}\\nDescription: {{issue['body']}}\\n"
        full_prompt = system_prompt + "\\n\\n" + issue_prompt
        
        print(f"Generating implementation plan for feature #{{issue['number']}}")
        print(f"Input prompt: {{issue_prompt}}")
        print(f"Input prompt length: {{len(full_prompt)}}")
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            temperature=0.2,
            top_p=0.75,
            top_k=40,
            num_beams=1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            encoder_repetition_penalty=1.0, 
            typical_p=1.0,
            length_penalty=1.2,
            do_sample=True,
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        response = response.strip()
        print("Final answer:", response)
        
        answers.append({{
            "issue_number": issue["number"],
            "issue_url": issue["url"],
            "answer": response,
            "generation_params": {{
                "temperature": 0.2,
                "max_new_tokens": 1024,
                "repetition_penalty": 1.2,
                "top_p": 0.75,
                "top_k": 40,
                "num_beams": 1,
                "no_repeat_ngram_size": 3,
                "encoder_repetition_penalty": 1.0,
                "typical_p": 1.0,
                "length_penalty": 1.2
            }}
        }})
        
    except Exception as e:
        print(f"Error generating answer for issue #{{issue.get('number', 'unknown')}}: {{e}}")
        continue

# Save answers
output_path = Path("/tmp/issue_answers")
output_path.mkdir(parents=True, exist_ok=True)

with (output_path / "answers.json").open("w") as f:
    json.dump({{
        "total_issues": len(issues),
        "total_answers": len(answers),
        "answers": answers,
        "timestamp": str(datetime.now().isoformat())
    }}, f, indent=2)

# Upload to GCS
for file_path in output_path.rglob("*"):
    if file_path.is_file():
        blob_path = f"llama/answers/{{file_path.relative_to(output_path)}}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(file_path))
"""
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={
                'memory': '16Gi',
                'cpu': '4',
                'nvidia.com/gpu': '1',
                'ephemeral-storage': '16Gi'
            },
            limits={
                'memory': '16Gi',
                'cpu': '4',
                'nvidia.com/gpu': '1',
                'ephemeral-storage': '16Gi'
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
        startup_timeout_seconds=600,
        **GCP_VOLUME_CONFIG
    )

def fetch_github_issues(
    dag: DAG,
    task_id: str,
    owner: str,
    repo: str,
    output_dir: str = "github_data",
) -> KubernetesPodOperator:
    """Fetch GitHub issues and discussions task"""
    return KubernetesPodOperator(
        task_id=task_id,
        dag=dag,
        name=task_id,
        namespace='airflow',
        image=BASE_IMAGE,
        cmds=["python", "-c"],
        arguments=[
            f"""\
import json
import os
from pathlib import Path
from google.cloud import storage
from datetime import datetime

# Import fetcher module
{FETCHER_CODE}

# Set up local directory
local_output_dir = Path("/tmp/{output_dir}")
local_output_dir.mkdir(parents=True, exist_ok=True)

# Fetch data from GitHub
github_token = '{Variable.get("github-token")}'
issues = fetch_github_issues("{owner}", "{repo}", github_token)
discussions = fetch_github_discussions("{owner}", "{repo}", github_token)

# Save data locally
save_to_dataset(issues, discussions, local_output_dir)

print(f"Fetched {{len(issues)}} issues and {{len(discussions)}} discussions")

# Upload to GCS
client = storage.Client()
bucket = client.bucket("{GCS_BUCKET}")

# Upload all files
for file_path in local_output_dir.rglob("*"):
    if file_path.is_file():
        blob_path = f"llama/github_data/{{file_path.relative_to(local_output_dir)}}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(file_path))

# Create metadata file
metadata = {{
    "timestamp": datetime.now().isoformat(),
    "repository": f"{owner}/{repo}",
    "issues_count": len(issues),
    "discussions_count": len(discussions),
    "files": [str(f.relative_to(local_output_dir)) for f in local_output_dir.rglob("*") if f.is_file()]
}}

metadata_path = local_output_dir / "metadata.json"
with metadata_path.open("w") as f:
    json.dump(metadata, f, indent=2)

# Upload metadata
blob = bucket.blob("llama/github_data/metadata.json")
blob.upload_from_filename(str(metadata_path))

print("Data upload complete")
"""
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={
                'memory': '4Gi',
                'cpu': '1',
                'ephemeral-storage': '8Gi'
            },
            limits={
                'memory': '4Gi',
                'cpu': '1',
                'ephemeral-storage': '8Gi'
            }
        ),
        is_delete_operator_pod=True,
        get_logs=True,
        startup_timeout_seconds=600,
        **GCP_VOLUME_CONFIG
    )

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
        "model_path": "meta-llama/Llama-3.1-8B-Instruct",
        'output_dir': MODEL_OUTPUT_PATH,
        "data_dir": DATASET_PATH,
        "checkpoint_dir": None,
        "num_epochs": 4,
        "batch_size": 4,
        "test_size": 0.001,
        "model_max_length": 1024,
        "seed": 41,
        "report_to": "wandb",
        "gradient_accumulation_steps": 8,
        "padding": "left",
        "dataloader_num_proc": 8,
        "use_fp16": True,
        "use_4bit": True,
        "use_qlora": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.05,
        "debug": False,
    }

    dataset_task = create_dataset(dag, "create_dataset")

    train_task = train_model(
        dag=dag,
        task_id="train",
        dataset_path=DATASET_PATH,
        config=config,
    )

    # 在现有的 DAG 定义中添加新的任务
    github_data = fetch_github_issues(
        dag=dag,
        task_id="fetch_github_data",
        owner="skypilot-org",
        repo="skypilot",
    )

    filtered_issues = filter_issues(
        dag=dag,
        task_id="filter_issues",
        dataset_path="llama/github_data",  # 更新为 github_data 的路径
        model_path="meta-llama/Llama-3.1-8B-Instruct",
    )

    answers = generate_answers(
        dag=dag,
        task_id="generate_answers",
        filtered_issues_path="llama/filtered_issues",
        base_model_path="meta-llama/Llama-3.1-8B-Instruct",
        adapter_path=f"llama/models/train",
    )

    # 更新任务依赖
    github_data >> filtered_issues >> answers
    dataset_task >> train_task >> answers
