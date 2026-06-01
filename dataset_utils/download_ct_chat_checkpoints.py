from huggingface_hub import hf_hub_download, snapshot_download


hf_token = ""
# hf_hub_download(repo_id='ibrahimhamamci/CT-RATE',
#     repo_type='dataset',
#     token=hf_token,
#     filename='models/RadBertClassifier.pth',
#     local_dir='<CT_CHAT_CHECKPOINTS_DIR>/', #'<CT_CHAT_CHECKPOINTS_DIR>/',
#     )

snapshot_download(
    repo_id="ibrahimhamamci/CT-RATE",
    repo_type="dataset",
    token=hf_token,
    allow_patterns="models/CT-CHAT/llava-lora-llama_3.1_8B/*",
    local_dir="<CT_CHAT_CHECKPOINTS_DIR>/",  #'<CT_CHAT_CHECKPOINTS_DIR>/',
)

# hf_hub_download(repo_id='ibrahimhamamci/CT-RATE',
#     repo_type='dataset',
#     token=hf_token,
#     filename='models/CT-CLIP-Related/CT-CLIP_v2.pt',
#     local_dir='<CT_CHAT_CHECKPOINTS_DIR>/', #'<CT_CHAT_CHECKPOINTS_DIR>/',
#     )
