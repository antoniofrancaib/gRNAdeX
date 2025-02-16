import os
import torch

""" Filter out sequences longer than 500 nucleotides for the sake of training the model with less GPU time
Output: Loaded 4223 entries from 'data/processed.pt'
Kept 3008 entries with <= 500 nucleotides (removed 1215)
Saved filtered dataset to 'data/processed_filtered.pt' """

DATA_PATH = "data"  
INPUT_FILE = "processed.pt"
OUTPUT_FILE = "processed_filtered.pt"

def main():
    # 1) Load the original processed dictionary
    input_path = os.path.join(DATA_PATH, INPUT_FILE)
    data_dict = torch.load(input_path)
    print(f"Loaded {len(data_dict)} entries from '{input_path}'")

    # 2) Filter out sequences longer than 500
    filtered_dict = {}
    for seq, info in data_dict.items():
        if len(seq) <= 500:
            filtered_dict[seq] = info

    print(f"Kept {len(filtered_dict)} entries with <= 500 nucleotides (removed {len(data_dict)-len(filtered_dict)})")

    # 3) Save the smaller dataset
    output_path = os.path.join(DATA_PATH, OUTPUT_FILE)
    torch.save(filtered_dict, output_path)
    print(f"Saved filtered dataset to '{output_path}'")

if __name__ == "__main__":
    main()
