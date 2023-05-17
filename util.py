from pathlib import Path


def output_directory(name):
    output_home = Path("results")
    output_directory = output_home / name
    output_directory.mkdir(parents=True, exist_ok=True)    
    return output_directory