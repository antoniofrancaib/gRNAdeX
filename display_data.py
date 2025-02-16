import os
import torch

# Set the path to your data directory
DATA_PATH = "data/" 

def load_and_print_data():
    # Load data from the processed.pt file
    data_list = list(torch.load(os.path.join(DATA_PATH, "das_split.pt")))

    # Print the first few entries to see how they look
    print("Sample data from data_list:")
    for i in range(min(5, len(data_list))):  # Print up to 5 samples
        print(f"Sample {i + 1}: {data_list[i]}")

if __name__ == "__main__":
    load_and_print_data()