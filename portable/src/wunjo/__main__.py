from wunjo.preload import main as preload
from wunjo.app import main as server

if __name__ == '__main__':
    preload()
    server()
