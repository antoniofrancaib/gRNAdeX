import pandas as pd
import numpy as np
import os
import ast
import argparse
import shutil

def create_folder(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def get_fasta_RNA2DNA(in_file, fasta_filename, out_dir):
    """ function that gets a dataframe with sequences and outputs their correspoding DNA
    sequences in fasta file format to operate with maw"""
    
    # make directory if it does not exist
    create_folder(out_dir)

    # read filename
    df = pd.read_csv(in_file)
    assert 'sequence' in df.columns, 'sequence column not found in dataframe'
    list_seq = df.sequence.tolist()
    # put in DNA format
    list_seq = [seq.replace('U', 'T').replace('_', 'N') for seq in list_seq]
    list_seq = [''.join(c for c in seq if c in 'ACGTN') for seq in list_seq]
    indices = [str(index) for index in df.index]

    with open(out_dir+fasta_filename, "w") as fasta_file:
        for name, sequence in zip(indices, list_seq):
            name = str(name)
            sequence = str(sequence)
            fasta_file.write(f">{name}\n")
            fasta_file.write(f"{sequence}\n")
    print(f"FASTA file '{fasta_filename}' has been created successfully in directory '{out_dir}.")


def create_individual_files(input_file, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    create_folder(output_dir)
    create_folder(output_dir+'csv_files/')
    create_folder(output_dir+'fasta_files/')

    df = pd.read_csv(input_file)
    rfam_list = df.rfam_list.tolist()
    unique_elements = set()
    for row in rfam_list:
        row_list = ast.literal_eval(row)
        for element in row_list:
            unique_elements.add(element)

    # list of unique rfam ids
    unique_elements_list = list(unique_elements)

    # create an individual file for each rfam id
    count_rows = 0
    for unique_rfam in unique_elements_list:
        # create a new dataframe with the info
        df_filtered = df[df['rfam_list'].apply(lambda x: unique_rfam in x)]
        count_rows += df_filtered.shape[0]
        filename = output_dir+'csv_files/'+unique_rfam+"_df.csv"
        df_filtered.to_csv(filename, index=False)
        print(f"CSV file {filename}' has been created successfully.")
        get_fasta_RNA2DNA(filename, unique_rfam+".fasta", output_dir+'fasta_files/')

    # we have to have more rows than in original dataframe (some sequences correspond to several rfams)
    assert count_rows >= df.shape[0], "Some sequences were lost during the filtering process."


def args_parser():
    parser = argparse.ArgumentParser(description='Create individual files for each rfam id')
    parser.add_argument('--input_file', type=str, required=True, help='Input RNA sequences CSV file with columns: sequence, rfam_list')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    return parser.parse_args()


if __name__ == "__main__":
    """ Code to create individual fasta files for each rfam id given a csv file with columns: sequence, rfam_list
    """
    args = args_parser()
    create_individual_files(args.input_file, args.output_dir)