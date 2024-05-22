import requests
import os
from tqdm import tqdm

# Base URL for the dataset
base_url = "https://physionet.org/files/icentia11k-continuous-ecg/1.0/"

# Directory to save downloaded files
download_dir = "/mnt/qnap/liranc6/data/icentia11k-continuous-ecg"

# Number of patients to download
num_patients = 500

# Create download directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Loop through patient IDs
for patient_id in tqdm(range(1, num_patients + 1), desc="Downloading patients"):
    # Format patient ID string
    patient_id_str = f"p{patient_id:05d}"
    patient_dir = f"p{patient_id_str[1:3]}"
    # Construct patient directory URL
    patient_dir_url = os.path.join(base_url, patient_dir, f"{patient_id_str}/")

    # Send GET request to patient directory
    response = requests.get(patient_dir_url)

    # Check for successful response
    if response.status_code == 200:
        # Extract file listings from HTML content
        file_list = response.text.split("<a href=\"")
        file_list = [f for f in file_list if f.startswith("p")]  # Filter relevant files

        # Download each file
        for filename in tqdm(file_list, desc=f"Downloading files for patient {patient_id_str}", leave=False):
            file_url = os.path.join(patient_dir_url, filename)
            file_path = os.path.join(download_dir, patient_id_str, filename)

            # Create patient subdirectory if needed
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Send GET request to download file
            file_response = requests.get(file_url)

            # Check for successful download
            if file_response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(file_response.content)
                tqdm.write(f"Downloaded {file_path}")
            else:
                tqdm.write(f"Failed to download {file_path} (status code: {file_response.status_code})")
    else:
        tqdm.write(f"Failed to access patient directory {patient_dir_url} (status code: {response.status_code})")

print("Download complete!")