import sys
from src.helper import Helper

if __name__ == "__main__":
    try:
        print(f"Purging checkpoints...")
        helper = Helper()
        helper.purge(ckpt=True)
    finally:
        print(f"Successfully purged the checkpoints!")