import sys

if sys.platform == 'win32':
    try:
        import torch
        from torch import nn
    except ImportError:
        # Torch is not installed or importable
        print("Torch is not defined")
        from wunjo.preload import main
    else:
        # Torch is already imported or available
        from wunjo.app import main
else:
    from wunjo.app import main


if __name__ == '__main__':
    main()
