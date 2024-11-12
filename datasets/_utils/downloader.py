import requests
import os
import zipfile
import tarfile
from tqdm import tqdm

def download_and_extract(name, url, extract_to='.'):
    try:
        # Step 1: Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        # Get the total file size from the headers
        total_size = int(response.headers.get('content-length', 0))

        # Save the file temporarily
        temp_file_path = os.path.join(extract_to, f'{name}_temp')
        with open(temp_file_path, 'wb') as file, tqdm(
            desc=f'Downloading {name}',
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                bar.update(len(chunk))

        # Step 2: Determine the file type
        with open(temp_file_path, 'rb') as file:
            # Read the first few bytes to determine the file type
            magic_bytes = file.read(262)

            if magic_bytes.startswith(b'PK'):
                file_extension = '.zip'
            elif magic_bytes.startswith(b'\x1f\x8b\x08'):
                file_extension = '.tar.gz'
            else:
                raise ValueError("Unsupported file type")

        # Step 3: Rename the temporary file with the correct extension
        file_path = os.path.join(extract_to, f'{name}{file_extension}')
        os.rename(temp_file_path, file_path)

        # Step 4: Decompress the file and avoid nested directories
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

# Example usage:
# download_and_extract('example', 'https://example.com/path/to/file')
