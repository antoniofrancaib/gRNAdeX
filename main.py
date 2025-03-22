import dotenv
dotenv.load_dotenv(".env")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import random
import argparse
import wandb
import numpy as np

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from src.trainer import train, evaluate
from src.data.dataset import RNADesignDataset, BatchSampler
from src.models import (
    AutoregressiveMultiGNNv1, 
    NonAutoregressiveMultiGNNv1,
)
from src.constants import DATA_PATH
from src.evaluator import evaluate

def set_seed(seed=0, device_type='cpu'):
    """
    Sets random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if device_type == 'xpu':
        import intel_extension_for_pytorch as ipex
        torch.xpu.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)

def load_avoid_sequences(config):
    """
    Load sequences to avoid from .fasta.out files in the processed_rfam_path directory.
    
    Args:
        config: Configuration object containing processed_rfam_path
        
    Returns:
        list: List of sequences to avoid
    """
    avoid_sequences = []
    rfam_dir = config.processed_rfam_path
    
    if not os.path.exists(rfam_dir):
        print(f"Warning: Directory {rfam_dir} does not exist. No sequences will be avoided.")
        return avoid_sequences
        
    for filename in os.listdir(rfam_dir):
        if filename.endswith('.fasta.out'):
            file_path = os.path.join(rfam_dir, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        continue
                    sequence = line.strip()
                    if sequence:  # Only add non-empty sequences
                        avoid_sequences.append(sequence)
    
    print(f"Loaded {len(avoid_sequences)} sequences to avoid from {rfam_dir}")
    return avoid_sequences


def main(config, device):
    """
    Main function for training and evaluating gRNAde.
    """
    # Set seed
    set_seed(config.seed, device.type)

    # Initialise model
    model = get_model(config).to(device)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'\nMODEL\n    {model}\n    Total parameters: {total_param}')
    wandb.run.summary["total_param"] = total_param

    # Load checkpoint
    if config.model_path != '':
        model.load_state_dict(torch.load(config.model_path, map_location=device))
    
    if config.evaluate:
        # Load test set
        _, _, test_list = get_data_splits(config, split_type=config.split)
        testset = get_dataset(config, test_list, split="test")
        test_loader = get_dataloader(config, testset, shuffle=False)

        # Load sequences to avoid if enabled
        avoid_sequences = None
        if config.avoid_sequences:
            avoid_sequences = load_avoid_sequences(config)

        # Run evaluator + save designed structures
        results = evaluate(
            model, 
            test_loader.dataset, 
            config.n_samples, 
            config.temperature, 
            device, 
            model_name="test",
            metrics=['recovery', 'perplexity', 'sc_score_eternafold', 'sc_score_ribonanzanet', 'sc_score_rhofold'],
            save_designs=True,
            avoid_sequences=avoid_sequences
        )
        df, samples_list, recovery_list, perplexity_list, \
        scscore_list, scscore_ribonanza_list, \
        scscore_rmsd_list, scscore_tm_list, scscore_gdt_list, \
        rmsd_within_thresh, tm_within_thresh, gdt_within_thresh = results.values()
        # Save results
        torch.save(results, os.path.join(wandb.run.dir, f"test_results.pt"))
        # Update wandb summary metrics
        wandb.run.summary[f"best_test_recovery"] = np.mean(recovery_list)
        wandb.run.summary[f"best_test_perplexity"] = np.mean(perplexity_list)
        wandb.run.summary[f"best_test_scscore"] = np.mean(scscore_list)
        wandb.run.summary[f"best_test_scscore_ribonanza"] = np.mean(scscore_ribonanza_list)
        wandb.run.summary[f"best_test_scscore_rmsd"] = np.mean(scscore_rmsd_list)
        wandb.run.summary[f"best_test_scscore_tm"] = np.mean(scscore_tm_list)
        wandb.run.summary[f"best_test_scscore_gdt"] = np.mean(scscore_gdt_list)
        wandb.run.summary[f"best_test_rmsd_within_thresh"] = np.mean(rmsd_within_thresh)
        wandb.run.summary[f"best_test_tm_within_thresh"] = np.mean(tm_within_thresh)
        wandb.run.summary[f"best_test_gdt_within_thresh"] = np.mean(gdt_within_thresh)
        print(f"BEST test recovery: {np.mean(recovery_list):.4f}\
                perplexity: {np.mean(perplexity_list):.4f}\
                scscore: {np.mean(scscore_list):.4f}\
                scscore_ribonanza: {np.mean(scscore_ribonanza_list):.4f}\
                scscore_rmsd: {np.mean(scscore_rmsd_list):.4f}\
                scscore_tm: {np.mean(scscore_tm_list):.4f}\
                scscore_gdt: {np.mean(scscore_gdt_list):.4f}\
                rmsd_within_thresh: {np.mean(rmsd_within_thresh):.4f}\
                tm_within_thresh: {np.mean(tm_within_thresh):.4f}\
                gdt_within_thresh: {np.mean(gdt_within_thresh):.4f}")

    else:
        # Get train, val, test data samples as lists
        train_list, val_list, test_list = get_data_splits(config, split_type=config.split)

        # Load datasets
        trainset = get_dataset(config, train_list, split="train")
        valset = get_dataset(config, val_list, split="val")
        testset = get_dataset(config, test_list, split="test")

        # Prepare dataloaders
        train_loader = get_dataloader(config, trainset, shuffle=True)
        val_loader = get_dataloader(config, valset, shuffle=False)
        test_loader = get_dataloader(config, testset, shuffle=False)
        
        # Run trainer
        train(config, model, train_loader, val_loader, test_loader, device)


def get_data_splits(config, split_type="structsim_v2"):
    """
    Returns train, val, test data splits as lists.
    """
    data_list = list(torch.load(os.path.join(DATA_PATH, "processed.pt")).values())
    
    def index_list_by_indices(lst, indices):
        # return [lst[index] if 0 <= index < len(lst) else None for index in indices]
        return [lst[index] for index in indices]
    
    # Pre-compute using notebooks/split_{split_type}.ipynb
    train_idx_list, val_idx_list, test_idx_list = torch.load(
        os.path.join(DATA_PATH, f"{split_type}_split.pt")) 
    train_list = index_list_by_indices(data_list, train_idx_list)
    val_list = index_list_by_indices(data_list, val_idx_list)
    test_list = index_list_by_indices(data_list, test_idx_list)

    return train_list, val_list, test_list


def get_dataset(config, data_list, split="train"):
    """
    Returns a Dataset for a given split.
    """
    # Load sequences to avoid if enabled
    avoid_sequences = None
    if config.avoid_sequences:
        avoid_sequences = load_avoid_sequences(config)

    return RNADesignDataset(
        data_list = data_list,
        split = split,
        radius = config.radius,
        top_k = config.top_k,
        num_rbf = config.num_rbf,
        num_posenc = config.num_posenc,
        max_num_conformers = config.max_num_conformers,
        noise_scale = config.noise_scale,
        avoid_sequences = avoid_sequences
    )


def get_dataloader(
        config, 
        dataset, 
        shuffle=True,
        pin_memory=True,
        exclude_keys=[],
    ):
    """
    Returns a DataLoader for a given Dataset.

    Args:
        dataset (RNADesignDataset): dataset object
        config (dict): wandb configuration dictionary
        shuffle (bool): whether to shuffle the dataset
        pin_memory (bool): whether to pin memory
        exclue_keys (list): list of keys to exclude during batching
    """
    return DataLoader(
        dataset, 
        num_workers = config.num_workers,
        batch_sampler = BatchSampler(
            node_counts = dataset.node_counts, 
            max_nodes_batch = config.max_nodes_batch,
            max_nodes_sample = config.max_nodes_sample,
            shuffle = shuffle,
        ),
        pin_memory = pin_memory,
        exclude_keys = exclude_keys
    )


def get_model(config):
    """
    Returns a Model for a given config.
    """
    model_class = {
        'ARv1' : AutoregressiveMultiGNNv1,
        'NARv1': NonAutoregressiveMultiGNNv1,
    }[config.model]
    
    return model_class(
        node_in_dim = tuple(config.node_in_dim),
        node_h_dim = tuple(config.node_h_dim), 
        edge_in_dim = tuple(config.edge_in_dim),
        edge_h_dim = tuple(config.edge_h_dim), 
        num_layers=config.num_layers,
        drop_rate = config.drop_rate,
        out_dim = config.out_dim
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='configs/default.yaml', type=str)
    parser.add_argument('--expt_name', dest='expt_name', default=None, type=str)
    parser.add_argument('--tags', nargs='+', dest='tags', default=[])
    parser.add_argument('--no_wandb', action="store_true")
    args, unknown = parser.parse_known_args()

    # Initialise wandb
    if args.no_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"), 
            entity=os.environ.get("WANDB_ENTITY"), 
            config=args.config, 
            name=args.expt_name, 
            mode='disabled'
        )
    else:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"), 
            entity=os.environ.get("WANDB_ENTITY"), 
            config=args.config, 
            name=args.expt_name, 
            tags=args.tags,
            mode='online'
        )
    config = wandb.config
    config_str = "\nCONFIG"
    for key, val in config.items():
        config_str += f"\n    {key}: {val}"
    print(config_str)

    # Set device (GPU/CPU/XPU)
    if config.device == 'xpu':
        import intel_extension_for_pytorch as ipex
        [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())]
        device = torch.device("xpu:{}".format(config.gpu) if torch.xpu.is_available() else 'cpu')
    else:
        device = torch.device("cuda:{}".format(config.gpu) if torch.cuda.is_available() else "cpu")
    
    # Run main function
    main(config, device)
