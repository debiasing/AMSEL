from os.path import isfile
from os import remove
import hashlib
from urllib.request import urlretrieve

class InvalidChecksumError(Exception):
    """Raised if a file's checksum does not match the expected value."""
    pass

def _calculate_checksum(filename: str, hash_algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    """Calculates the checksum of a file."""
    if not isfile(filename):
        raise FileNotFoundError(f"File to hash could not be located: {filename}")

    # Use hashlib.new() to create a hash object from the algorithm's name
    file_hash = hashlib.new(hash_algorithm)
    
    with open(filename, "rb") as f:
        # Read the file in chunks to conserve memory
        while chunk := f.read(chunk_size):
            file_hash.update(chunk)
            
    return file_hash.hexdigest()

def download_and_verify(
    url: str,
    filename: str,
    expected_checksum: str,
    hash_algorithm: str = "sha256"
):
    """Downloads a file and verifies its integrity using a checksum.

    This function provides an atomic download-and-verify operation. It ensures
    that if the download or verification fails for any reason, no partial or
    corrupt file will be left at the destination path.

    Args:
        url (str): The source URL of the file to download.
        filename (str): The local path where the file will be saved.
        expected_checksum (str): The expected hash digest of the file for
                                 verification.
        hash_algorithm (str, optional): The name of the hash algorithm to use,
                                        e.g., 'sha256' or 'md5'. Defaults to
                                        'sha256'.

    Raises:
        InvalidChecksumError: If the calculated checksum of the downloaded file
                              does not match the expected_checksum.
        Exception: Re-raises exceptions from the underlying network or file
                   system operations if the download fails.
    """
    try:
        # 1. Download the file.
        print(f"Downloading {url} to {filename}...")
        urlretrieve(url=url, filename=filename)

        # 2. Calculate the checksum of the downloaded file.
        print("Verifying checksum...")
        calculated_checksum = _calculate_checksum(filename, hash_algorithm)

        # 3. Compare checksums and raise an error on mismatch.
        # Checksums provided by the user might be upper/lower case, so we 
        # perform a case-insensitive comparison.
        if calculated_checksum.lower() != expected_checksum.lower():
            raise InvalidChecksumError(
                f"Checksum mismatch for {filename}. Please retry or check repository for updates."
            )
    except Exception:
        if isfile(filename):
            print(f"An error occurred. Deleting incomplete/corrupt file: {filename}")
            remove(filename)
        # Re-raise the exception so the caller knows the operation failed
        raise