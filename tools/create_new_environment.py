import os

if __name__ == "__main__":
    try:
        environment_name = input("Please enter the name of the environment: ")
        print(f"Creating a new environment \"{environment_name}\"...")
        os.system(f"conda create --name {environment_name} --file requirements.txt")
    finally:
        print(f"Successfully created a new environment!")
