from datetime import datetime
import os
from pathlib import Path
import sys
import importlib.metadata


def output_directory(tag=None):
    output_home = Path("results")
    now = datetime.now().strftime("%d-%m-%y@%H:%M:%S")
    if tag is not None:
        output_directory_name = f"{tag}-{now}"
    else:
        output_directory_name = now
    output_directory = output_home / output_directory_name
    output_directory.mkdir(parents=True, exist_ok=True)
    
    dump_details(output_directory)
    
    return output_directory


def dump_details(output_directory):
    root = Path(os.environ["CODEROOT"]).resolve()
    all_modules = set()
    for name in sys.modules:
        module = sys.modules[name]
        try:
            module_file = Path(module.__file__)
            is_descendant = root in module_file.resolve().parents
            if is_descendant:
                all_modules.add(module_file.resolve())
        except (AttributeError, TypeError):
            pass

    Path(output_directory / "details").mkdir(exist_ok=True, parents=True)

    code_directory = Path(output_directory) /  "details" / "code"
    for module_path in all_modules:
        target_path = code_directory / module_path.relative_to(root)
        target_path.parent.mkdir(exist_ok=True, parents=True)
        with open(target_path, "w") as file, open(module_path, "r") as source_file:
            file.write(source_file.read())

    with open(Path(output_directory) / "details"  / "run.sh", "w") as file:
        print("python", *sys.argv, file=file)

    with open(Path(output_directory) / "details" / "requirements.txt", "w") as file:
        packages = importlib.metadata.distributions()
        for package in packages:
            name = package.metadata["Name"]
            version = package.metadata["Version"]
            print(f"{name}=={version}", file=file)


class OutputWriter:

    def __init__(self, filepath, debug=False):
        self.filepath = filepath
        self.debug = debug
        self.started = False
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)

    def write(self, string):
        with open(self.filepath, "ab") as output_file:
            buffer = string.encode("utf-8")
            if self.started:
                output_file.write(b"\0")
            output_file.write(buffer)
            output_file.flush()

        if self.debug:
            if self.started:
                print()
                print("-" * 80)
                print()
            print(string)

        self.started = True