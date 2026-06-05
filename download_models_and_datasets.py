
import nltk
nltk.download('punkt_tab')

from datasets import load_dataset
from huggingface_hub import hf_hub_download

load_dataset("dgslibisey/MuSiQue")
load_dataset("mandarjoshi/trivia_qa","rc")

for split, filename in [("train", "train.parquet"), ("dev", "dev.parquet"), ("test", "test.parquet")]:
    local_path = hf_hub_download(
        repo_id="xanhho/2WikiMultihopQA",
        filename=filename,
        repo_type="dataset",
    )
    ds_split = load_dataset("parquet", data_files={split: local_path}, split=split)

from huggingface_hub import snapshot_download
snapshot_download("ibm-granite/granite-4.1-3b")
snapshot_download("ibm-granite/granite-4.1-8b")
snapshot_download("Qwen/Qwen3-4B-Instruct-2507")
snapshot_download("meta-llama/Llama-3.1-8B-Instruct")
