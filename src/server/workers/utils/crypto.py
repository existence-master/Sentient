import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

from dotenv import load_dotenv

# Load .env file for 'dev-local' environment.
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
AES_SECRET_KEY_HEX = os.getenv("AES_SECRET_KEY")
AES_IV_HEX = os.getenv("AES_IV")

AES_SECRET_KEY = None
AES_IV = None

if AES_SECRET_KEY_HEX:
    if len(AES_SECRET_KEY_HEX) == 64:  # 32 bytes = 64 hex chars
        AES_SECRET_KEY = bytes.fromhex(AES_SECRET_KEY_HEX)
    else:
        print(f"[Worker_Crypto_WARNING] AES_SECRET_KEY is invalid. Encryption/Decryption will fail.")
else:
    print(f"[Worker_Crypto_WARNING] AES_SECRET_KEY is not set. Encryption/Decryption will fail.")

if AES_IV_HEX:
    if len(AES_IV_HEX) == 32:  # 16 bytes = 32 hex chars
        AES_IV = bytes.fromhex(AES_IV_HEX)
    else:
        print(f"[Worker_Crypto_WARNING] AES_IV is invalid. Encryption/Decryption will fail.")
else:
    print(f"[Worker_Crypto_WARNING] AES_IV is not set. Encryption/Decryption will fail.")


def aes_encrypt(data: str) -> str:
    if not AES_SECRET_KEY or not AES_IV:
        raise ValueError("AES encryption keys are not configured for worker.")
    backend = default_backend()
    cipher = Cipher(algorithms.AES(AES_SECRET_KEY), modes.CBC(AES_IV), backend=backend)
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data.encode()) + padder.finalize()
    encrypted = encryptor.update(padded_data) + encryptor.finalize()
    return base64.b64encode(encrypted).decode()


def aes_decrypt(encrypted_data: str) -> str:
    if not AES_SECRET_KEY or not AES_IV:
        raise ValueError("AES encryption keys are not configured for worker.")
    backend = default_backend()
    cipher = Cipher(algorithms.AES(AES_SECRET_KEY), modes.CBC(AES_IV), backend=backend)
    decryptor = cipher.decryptor()
    encrypted_bytes = base64.b64decode(encrypted_data)
    decrypted = decryptor.update(encrypted_bytes) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    unpadded_data = unpadder.update(decrypted) + unpadder.finalize()
    return unpadded_data.decode()
