import gc
import torch


class ModelCache:
    def __init__(self, max_models=5):
        self.cache = {}
        self.max_models = max_models

    def clear(self):
        if len(self.cache) > 0:
            print("Clear model cache")
            self.cache = {}

            if torch.cuda.is_available():
                for device_id in range(torch.cuda.device_count()):
                    torch.cuda.set_device(device_id)
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            gc.collect()

    def get_model(self, model_name):
        if model_name in self.cache:
            print(f"Using cached model: {model_name}")
            return self.cache[model_name]
        return None

    def update_model(self, model_name, loaded_model):
        if len(self.cache) >= self.max_models:
            self.evict_model()

        self.cache[model_name] = loaded_model

    def evict_model(self):
        # Implement LRU or other eviction policy (FIFO here)
        model_name = next(iter(self.cache))  # Simple FIFO eviction
        print(f"Evicting model: {model_name}")
        del self.cache[model_name]
