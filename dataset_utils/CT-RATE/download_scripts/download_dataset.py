import shutil
import os
import pandas as pd

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from dotenv import load_dotenv

# Allows to automatically load environment variables from a .env file
# Save locally .env file with your token as HF_TOKEN=your_token
load_dotenv()

split = "train_fixed"
batch_size = 10
start_at = 20099
k = 40000

repo_id = "ibrahimhamamci/CT-RATE"
directory_name = f"dataset/{split}/"
hf_token = os.getenv("HF_TOKEN")

data = pd.read_csv(
    f"<CT_RATE_DATASET_DIR>/train_labels.csv"
)

for i in tqdm(range(start_at, min(len(data), k), batch_size)):

    data_batched = data[i : i + batch_size]

    for name in data_batched["VolumeName"]:
        folder1 = name.split("_")[0]
        folder2 = name.split("_")[1]
        folder = folder1 + "_" + folder2
        folder3 = name.split("_")[2]
        subfolder = folder + "_" + folder3
        subfolder = directory_name + folder + "/" + subfolder

        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
            subfolder=subfolder,
            filename=name,
            #            cache_dir='',
            local_dir="<CT_RATE_DATASET_DIR>/train_fixed",
            local_dir_use_symlinks=False,
            resume_download=True,
        )

#    shutil.rmtree('./datasets--ibrahimhamamci--CT-RATE')
