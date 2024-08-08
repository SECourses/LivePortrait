from huggingface_hub import HfApi
import os
import requests
from tqdm import tqdm
import time

# Repository details
repo_id = "OwlMaster/LivePortrait"
repo_type = "model"

# Initialize Hugging Face API
api = HfApi()

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def download_file(file, local_path):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{file}"
    
    # Check file size on server
    response = requests.head(url, allow_redirects=True)
    server_file_size = int(response.headers.get('content-length', 0))
    
    # Check if file exists locally and compare sizes
    if os.path.exists(local_path):
        local_file_size = os.path.getsize(local_path)
        if local_file_size == server_file_size:
            print(f"File already exists and has correct size: {local_path}")
            return True
        elif local_file_size > server_file_size:
            print(f"Local file is larger than server file. Redownloading: {local_path}")
            os.remove(local_path)
        else:
            print(f"Resuming download for: {local_path}")
    else:
        print(f"Starting new download: {local_path}")

    # Determine the starting position for resume
    initial_pos = os.path.getsize(local_path) if os.path.exists(local_path) else 0
    mode = 'ab' if initial_pos > 0 else 'wb'
    
    headers = {'Range': f'bytes={initial_pos}-'} if initial_pos > 0 else {}

    max_retries = 5
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()
            total_size = server_file_size

            with open(local_path, mode) as f, tqdm(
                desc=file,
                initial=initial_pos,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        progress_bar.update(size)

            actual_size = os.path.getsize(local_path)
            if actual_size == server_file_size:
                print(f"Successfully downloaded: {local_path}")
                print(f"Size: {format_size(actual_size)}")
                return True
            else:
                print(f"Size mismatch for {file}. Expected: {server_file_size}, Actual: {actual_size}")
                if attempt < max_retries - 1:
                    print(f"Retrying download (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(retry_delay)
                    # Update initial position and mode for next attempt
                    initial_pos = actual_size
                    mode = 'ab'
                    headers = {'Range': f'bytes={initial_pos}-'}
                else:
                    return False

        except requests.RequestException as e:
            print(f"Error downloading {file}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying download (attempt {attempt + 2}/{max_retries})...")
                time.sleep(retry_delay)
                # Update initial position and mode for next attempt
                initial_pos = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                mode = 'ab' if initial_pos > 0 else 'wb'
                headers = {'Range': f'bytes={initial_pos}-'} if initial_pos > 0 else {}
            else:
                return False

    return False

# Get list of files in the repository
files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

# Download each file
for file in files:
    local_path = os.path.join(os.getcwd(), file)
    ensure_dir(local_path)

    if download_file(file, local_path):
        print(f"File verified: {file}")
    else:
        print(f"Failed to download or verify: {file}")

print("Download complete.")