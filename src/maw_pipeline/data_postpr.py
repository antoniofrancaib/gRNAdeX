import os
import argparse
import glob

# convert again to RNA format
def convert_back_to_RNA(seq):
    seq = seq.replace('T', 'U').replace('N', '_')
    seq = ''.join(c for c in seq if c in 'ACGU_')
    return seq

def process_maw_results(input_dir, output_dir, k_min, k_max):
    """
    Process MAW results from .fasta.out files:
    - Filter sequences based on length (k_min ≤ length ≤ k_max)
    - Combine unique sequences from each file
    - Output results in FASTA format with filename as header
    
    Args:
        input_dir: Directory containing .fasta.out files
        output_dir: Directory to save processed results
        k_min: Minimum sequence length to keep
        k_max: Maximum sequence length to keep
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all .fasta.out files in the input directory
    files = glob.glob(os.path.join(input_dir, "*.fasta.out"))
    
    if not files:
        print(f"No .fasta.out files found in {input_dir}")
        return
    
    # Process each file
    for file_path in files:
        file_name = os.path.basename(file_path)
        output_file = os.path.join(output_dir, file_name)
        
        # Get header name from filename (removing .fasta.out extension)
        header_name = file_name.replace(".fasta.out", "")
        
        # Set to store unique sequences
        unique_sequences = set()
        
        # Current sequence header
        current_header = None
        
        # Read the file and process sequences
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    # New sequence header found
                    current_header = line
                elif line and current_header is not None:
                    # This is a sequence line
                    sequence = line.strip()
                    seq_len = len(sequence)
                    sequence = convert_back_to_RNA(sequence)
                    
                    # Filter based on length
                    if k_min <= seq_len <= k_max:
                        unique_sequences.add(sequence)
        
        # Write filtered unique sequences to output file
        with open(output_file, 'w') as out_file:
            # Write header
            out_file.write(f">{header_name}\n")
            
            # Write unique sequences
            for seq in sorted(unique_sequences):
                out_file.write(f"{seq}\n")
        
        print(f"Processed {file_name} - Found {len(unique_sequences)} unique sequences with length between {k_min} and {k_max}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MAW results and filter sequences by length.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing .fasta.out files")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save processed results")
    parser.add_argument("-k", "--k_min", type=int, default=11, help="Minimum sequence length to keep")
    parser.add_argument("-K", "--k_max", type=int, default=16, help="Maximum sequence length to keep")
    
    args = parser.parse_args()
    
    process_maw_results(args.input_dir, args.output_dir, args.k_min, args.k_max)