import re
import sys
import subprocess
from pathlib import Path
from packaging import version

try:
    nvcc_version = subprocess.check_output(["nvcc", "--version"], universal_newlines=True)
    nvcc_version = re.search(r"release (\d+\.\d+)", nvcc_version).group(1) if nvcc_version else None
    if sys.platform == 'win32':  # Access on win32 install CPU version for old CUDA
        if version.parse(nvcc_version) < version.parse("11.8"):  # Min CUDA version
            nvcc_version = None
except (subprocess.CalledProcessError, FileNotFoundError):
    nvcc_version = None

if sys.platform == 'win32':
    try:
        import torch
        from torch import nn
        import onnxruntime

        torch_version = torch.__version__

        # Old torch version
        if version.parse(torch_version) < version.parse("2.2.2"):
            print(f"Torch version {torch_version} is below 2.2.2")
            from wunjo.preload import main
        else:
            print(f"Torch version {torch_version} is 2.2.2 or higher")

            cuda_version = torch.version.cuda

            if cuda_version is None and nvcc_version is None:
                print(f"CPU version")
                from wunjo.app import main
            elif cuda_version is not None and nvcc_version is not None and version.parse(nvcc_version) >= version.parse(cuda_version):
                print(f"CUDA version {cuda_version}")
                from wunjo.app import main
            else:
                print("CUDA version is below minimum requirements")
                from wunjo.preload import main
    except ImportError:
        # Torch is not installed or importable
        print("Torch is not defined")
        from wunjo.preload import main
else:
    import torch

    cuda_version = torch.version.cuda  # For linux will need to control to torch if with cuda when cuda version of nvcc equal

    if nvcc_version is None:
        # Check if there is a 'cuda' directory with version info in `/usr/local`
        for path in Path("/usr/local").glob("cuda*"):
            match = re.search(r"cuda-(\d+\.\d+)", str(path))
            if match:
                if version.parse(cuda_version) <= version.parse(match.group(1)):
                    nvcc_version = match.group(1)
                    break

    if cuda_version is None and nvcc_version is None:
        print(f"CPU version")
        from wunjo.app import main
    elif cuda_version is not None and nvcc_version is not None and version.parse(nvcc_version) >= version.parse(cuda_version):
        print(f"CUDA version {cuda_version}")
        from wunjo.app import main
    else:
        print("CUDA version is below minimum requirements")
        from wunjo.preload import main

if __name__ == '__main__':
    main()
