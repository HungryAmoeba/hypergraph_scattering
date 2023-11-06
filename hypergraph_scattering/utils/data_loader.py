import os
import yaml

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_data(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)
        # Implement your data loading logic here
        # For example, load a CSV file, an image, etc.
        return data

def get_data_loader():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    data_dir = config.get("data_directory", "../data")
    return DataLoader(data_dir)

