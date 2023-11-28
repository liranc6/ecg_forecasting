# README for the icentia11k_make_wfdb.py script

#
# iCentia11k to WFDB Conversion Script

This Python script is designed to convert physiological data from the iCentia11k dataset to the WFDB (WaveForm DataBase) format. The conversion involves processing electrocardiogram (ECG) data and creating WFDB files for each subject.

## Script Overview

### Label Mapping

The script includes a `label_mapping` dictionary that maps integer labels to symbols and descriptions for different beat and rhythm types.

### `get_person_attributes` Function

Extracts information (index, data path, labels path) from the given file path.

### `make_wfdb` Function

1. Reads iCentia11k data and labels from pickle files.
2. Converts the data to WFDB format using the `wfdb.io.wrsamp` and `wfdb.io.wrann` functions.
3. Handles the conversion of beat and rhythm annotations.

### `validate_wfdb` Function

Validates the created WFDB files against the original iCentia11k data, comparing signal sizes, beat and rhythm annotations, and checking for discrepancies.

### Workflow

1. The script processes each subject's data individually.
2. Creates a WFDB directory (`p{person_idx:05d}`) for each subject.
3. Extracts beat and rhythm annotations and converts them to WFDB format.
4. Validates the created WFDB files against the original data.

### Execution

The script demonstrates its functionality by calling `make_wfdb` with a specific iCentia11k data file (`06039_batched.pkl.gz`).

Note: The script assumes a specific directory structure for the output WFDB files and uses the `wfdb` library for WFDB file creation and validation.

Feel free to reach out if you have any questions or need further clarification!
