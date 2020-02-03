import sys
from src.helper import Helper

if __name__ == "__main__":
    model_name = sys.argv[1]
    try:
        print(f"Purging model\"{model_name}\"...")
        helper = Helper()
        helper.purge(model_name)
    finally:
        print(f"Successfully purged the model!")