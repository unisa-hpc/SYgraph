import requests
import os
import zipfile
import tarfile

import requests
import os
import zipfile
import tarfile
import shutil
import gdown

def check_disk_space(directory, required_space):
    """Check if there's enough space on the drive."""
    total, used, free = shutil.disk_usage(directory)
    return free >= required_space

def download_and_extract(name, url, download_dir='.', extract_to='.'):
    try:
        # Ensure download directory exists
        os.makedirs(download_dir, exist_ok=True)
        
        # Step 1: Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        # Get the total file size from the headers
        total_size = int(response.headers.get('content-length', 0))
        
        # Check disk space before downloading
        if not check_disk_space(download_dir, total_size):
            raise OSError("Not enough space in the specified download directory.")

        # Save the file temporarily in the specified download directory
        temp_file_path = os.path.join(download_dir, f'{name}_temp')
        gdown.download(url, temp_file_path, quiet=False)

        # Step 2: Determine the file type
        with open(temp_file_path, 'rb') as file:
            magic_bytes = file.read(262)
            if magic_bytes.startswith(b'PK'):
                file_extension = '.zip'
            elif magic_bytes.startswith(b'\x1f\x8b\x08'):
                file_extension = '.tar.gz'
            else:
                raise ValueError("Unsupported file type")

        # Step 3: Rename the temporary file with the correct extension
        file_path = os.path.join(download_dir, f'{name}{file_extension}')
        os.rename(temp_file_path, file_path)

        # Step 4: Decompress the file and avoid nested directories
        os.makedirs(extract_to, exist_ok=True)  # Ensure extraction directory exists
        if file_extension == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    member_path = os.path.join(extract_to, os.path.basename(member))
                    if member.endswith('/'):
                        continue  # Skip folders
                    with zip_ref.open(member) as source, open(member_path, "wb") as target:
                        target.write(source.read())
        elif file_extension == '.tar.gz':
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                for member in tar_ref.getmembers():
                    if member.isdir():
                        continue  # Skip folders
                    member_path = os.path.join(extract_to, os.path.basename(member.name))
                    with tar_ref.extractfile(member) as source, open(member_path, "wb") as target:
                        target.write(source.read())

        # Step 5: Clean up the downloaded archive file
        os.remove(file_path)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the file: {e}")
    except (zipfile.BadZipFile, tarfile.TarError) as e:
        print(f"An error occurred while extracting the file: {e}")
    except ValueError as e:
        print(e)
    except OSError as e:
        print(f"Error with disk space or directory: {e}")
