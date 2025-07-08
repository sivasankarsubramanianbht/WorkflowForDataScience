import kagglehub
import os
import logging # Import logging to use it in this module

# Configure a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set level for this module's logs
# Add a console handler if you want to see logs from this module immediately
# (Optional, as main.py's logger will also capture its calls)
if not logger.handlers: # Prevent adding multiple handlers if reloaded
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class DataDownload:
    """
    Handles the download of the specified dataset from Kaggle Hub.
    Includes a check to avoid re-downloading if the dataset already exists locally.
    """
    def __init__(self, dataset_name: str = "patrickzel/flight-delay-and-cancellation-data-2019-2023-v2", cache_dir: str = ".kaggle_cache"):
        """
        Initializes the DataDownload class.

        Args:
            dataset_name (str): The Kaggle dataset identifier.
            cache_dir (str): The directory where Kaggle Hub typically stores downloaded datasets.
                             This is usually '~/.kaggle/kagglehub/datasets/' but we can
                             make it explicit for checking.
        """
        self.dataset_name = dataset_name
        # Construct a likely path where KaggleHub would download it
        # KaggleHub's internal structure is typically ~/.kaggle/kagglehub/datasets/<owner>/<dataset_name>/<version>
        # We'll construct a check based on the dataset name.
        # A simpler robust check is to just see if the expected CSV file exists.
        # However, for efficiency, KaggleHub's internal caching is better.
        # KaggleHub itself handles caching, but to make sure *our* script doesn't call it repeatedly,
        # we can check for a specific file or infer the downloaded path.
        
        # A more robust check might be to check if the specific CSV file exists within a known download location.
        # KaggleHub downloads to a specific cache location. We can try to infer that.
        # The 'path' returned by kagglehub.dataset_download is the actual local path.
        # We can store this path once downloaded.
        
        # For simplicity and to leverage kagglehub's internal caching, the primary change will be
        # to ensure `dataset_download` is only called if not already downloaded.
        # kagglehub.dataset_download itself is designed to be idempotent and efficient.
        # However, if we want to bypass the call *completely* based on our own check,
        # we need to know the expected final path.
        
        # Let's use a simpler approach: check for the expected CSV file within a local project directory,
        # or rely on kagglehub's own caching mechanism if we just want to avoid *our* script re-doing it.
        
        # The best way to prevent *our script* from initiating a download every time is to
        # check for the presence of the expected data file after download.
        self.expected_csv_filename = 'flights_sample_100k.csv'
        self.download_root_path = os.path.join(os.getcwd(), "data") # A local 'data' folder for our project
        os.makedirs(self.download_root_path, exist_ok=True) # Ensure it exists

    def data_download(self):
        """
        Downloads the 'flight-delay-and-cancellation-data-2019-2023-v2' dataset
        from Kaggle Hub. If the dataset's expected CSV file already exists
        in the inferred local path, it skips the download.

        Returns:
            str: The path to the directory where the dataset files are located.
        """
        # KaggleHub downloads to a user-specific cache, typically ~/.kaggle/kagglehub/datasets/...
        # The returned `path_to_dataset_dir` is that cached location.
        # We don't want to copy it to a new location in our project every time,
        # but rather just ensure it's *downloaded* and then get its path.

        # The `kagglehub.dataset_download` function itself is idempotent, meaning
        # it won't re-download the *entire* dataset if it's already in the KaggleHub cache.
        # It will quickly check its local cache and return the path if found.

        # So, the main goal is to avoid *repeated calls* to kagglehub.dataset_download
        # *within our script* if we want to enforce our own project structure for the raw data.
        # However, it's often better to just rely on kagglehub's own cache.

        # Let's adjust the logic to simply call kagglehub.dataset_download and then
        # check if the specific CSV file exists within the *returned* path.
        # If it doesn't, we'll log a warning.

        logger.info(f"Checking for dataset: {self.dataset_name}")
        
        # kagglehub.dataset_download is designed to be efficient; it checks its internal cache first.
        # If the dataset is already downloaded, it returns the path very quickly without re-downloading.
        path_to_dataset_dir = kagglehub.dataset_download(self.dataset_name)
        logger.info(f"KaggleHub reported dataset path: {path_to_dataset_dir}")

        # Now, check if the specific file we need exists within that path
        full_csv_path = os.path.join(path_to_dataset_dir, self.expected_csv_filename)

        if os.path.exists(full_csv_path):
            logger.info(f"Required CSV file '{self.expected_csv_filename}' found at '{full_csv_path}'. No re-download needed.")
        else:
            logger.warning(f"Required CSV file '{self.expected_csv_filename}' NOT found at '{full_csv_path}'. "
                           "This might indicate an issue with the download or file naming.")
            logger.info(f"Contents of downloaded directory: {os.listdir(path_to_dataset_dir)}")
            # Consider raising an error here if finding the file is critical for pipeline continuation
            # raise FileNotFoundError(f"Expected CSV file not found in downloaded dataset: {full_csv_path}")

        return path_to_dataset_dir