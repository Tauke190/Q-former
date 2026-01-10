from huggingface_hub import snapshot_download

# Specify the dataset repo ID and the local directory where you want to download
repo_id = "liuhaotian/LLaVA-Instruct-150K"  # Replace with the actual dataset repo ID
local_dir = "data"  # Replace with your desired directory

# Download the dataset snapshot
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False  # Set to False to copy files instead of symlinking
)

print(f"Dataset downloaded to: {local_dir}")