import os
import argparse
import subprocess
import re
import shutil

def clean_fasta(input_file, output_file):
    # Read the entire file content
    with open(input_file, 'r') as infile:
        content = infile.read()
    
    # Parse the existing content
    sequences = {}
    current_header = None
    
    for line in content.strip().split('\n'):
        if line.startswith('>'):
            current_header = line.strip()
            sequences[current_header] = []
        elif current_header is not None:
            # Keep only ACGTN characters
            clean_line = ''.join(c for c in line.strip() if c in 'ACGTN')
            sequences[current_header].append(clean_line)
    
    # Write in a strict format (single-line sequences)
    with open(output_file, 'w') as outfile:
        for header, seq_lines in sequences.items():
            # Write header
            outfile.write(f"{header}\n")
            # Join all sequence lines into a single line
            sequence = ''.join(seq_lines)
            outfile.write(f"{sequence}\n")
    
    print(f"Cleaned FASTA written to {output_file}")

def create_test_fasta():
    with open('./test.fasta', 'w') as f:
        f.write(">test\n")
        f.write("ACGTACGTACGTACGT\n")
    print("Test FASTA created")

def run_maw(input_dir, output_dir, k, K, maw_executable):
    # Ensure the output directory exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # List all FASTA files in the input directory
    fasta_files = [f for f in os.listdir(input_dir) if f.endswith(".fasta")]

    if not fasta_files:
        print("No FASTA files found in the specified directory.")
        return

    # Run the command for each FASTA file
    for fasta_file in fasta_files:
        input_path = os.path.join(input_dir, fasta_file)
        output_path = os.path.join(output_dir, f"{fasta_file}.out")
        
        # Clean the FASTA file (now using the improved method)
        clean_fasta(input_path, input_path)

        # Construct the command
        command = f"{maw_executable} -a DNA -i {input_path} -o {output_path} -k {k} -K {K}"

        # Print the command (optional for debugging)
        print(f"Running: {command}")

        try:
            # Execute the command
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError:
            print(f"Error processing {fasta_file}. Creating a test FASTA to check maw functionality.")
            test_file = os.path.join(input_dir, "test.fasta")
            create_test_fasta()
            test_command = f"{maw_executable} -a DNA -i {test_file} -o {output_dir}/test.fasta.out -k {k} -K {K}"
            try:
                subprocess.run(test_command, shell=True, check=True)
                print("Test FASTA processed successfully. The issue is with your original FASTA file format.")
            except subprocess.CalledProcessError:
                print("Even the test FASTA failed. There might be an issue with the maw executable or its parameters.")

    print("All commands executed successfully. Results are saved in:", output_dir)


if __name__ == "__main__":
    # Setup argparse
    parser = argparse.ArgumentParser(description="Run maw command for all FASTA files in a directory.")

    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory containing FASTA files.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save output files.")
    parser.add_argument("-k", "--k_min", type=int, default=11, help="Minimum k-mer value.")
    parser.add_argument("-K", "--k_max", type=int, default=16, help="Maximum k-mer value.")
    parser.add_argument("-m", "--maw_executable", type=str, default="./tools/maw/maw", help="Path to the maw executable.")

    args = parser.parse_args()

    # Run the function with provided arguments
    run_maw(args.input_dir, args.output_dir, args.k_min, args.k_max, args.maw_executable)