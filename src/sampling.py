################################################################
# Generalisation of codes found online
# for explicit multi-state biomolecule representation learning.
# Original repositories:
# https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
################################################################

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

################################################################

def choose_nts(lgts, strategy='categorical', beam_branch=2, top_k=2, top_p=0.9, min_p=0.05, temperature=0.1):
    """
    lgts: tensor with shape batch_size, vocab_size
    samples according to beam_branch (will establish number of samples obtained)
    Returns: sample from the next_token probabilities
    """
    # First rescale with temperature -- is this right ? or should I rescale after filtering ?
    lgts = lgts / temperature

    if strategy.lower() == 'categorical':
        # Original code
        next_token_probs = F.softmax(lgts, dim=-1) ## WOULD THIS BE CORRECT ? Debug later -- which to use ?
        # change so it can return 2 samples if needed!
        #sample = Categorical(logits=lgts).sample()
        sample = torch.multinomial(next_token_probs, num_samples=beam_branch, replacement=True)
        return sample, next_token_probs

    elif strategy.lower() == 'top_k':
        # top-k logic
        filtered_lgts = top_k_filtering(lgts, top_k)
        next_token_probs = F.softmax(filtered_lgts, dim=-1)
        sample = torch.multinomial(next_token_probs, num_samples=beam_branch, replacement=True)
        return sample, next_token_probs
    
    elif strategy.lower() == 'top_p':
        # top-p logic
        filtered_lgts = top_p_filtering(lgts, top_p)
        next_token_probs = F.softmax(filtered_lgts, dim=-1)
        sample = torch.multinomial(next_token_probs, num_samples=beam_branch, replacement=True)
        return sample, next_token_probs

    elif strategy.lower() == 'min_p':
        # min-p logic
        filtered_lgts = min_p_sampling(lgts, min_p)
        next_token_probs = F.softmax(filtered_lgts, dim=-1)
        sample = torch.multinomial(next_token_probs, num_samples=beam_branch, replacement=True)

        return sample, next_token_probs
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    

def top_k_filtering(logits, top_k=2, filter_value=-float('Inf')):
    # Code from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317 -- slightly modified
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]

        logits[indices_to_remove] = filter_value
    return logits


def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    # Code from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317 
    # and https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
    # Slightly modified

    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        sorted_logits[sorted_indices_to_remove] = filter_value
        logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
    return logits


def min_p_sampling(logits, min_p=0.05, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using min-p
        Args:
            logits: logits distribution shape (vocabulary size)
            min_p >0.0: keep the tokens with probabilities greater than the probability
            scaled threshold determined after multiplying min_p by the maximum probabilities.
            (https://arxiv.org/html/2407.01082v1#S3)
    """
    # Convert logits to probs
    next_token_probs = F.softmax(logits, dim=-1)
    max_probs, _ = next_token_probs.max(dim=-1, keepdim=True)
    min_probs = min_p * max_probs  # Shape: (batch_size, res_len, 1)
    logits[next_token_probs < min_probs] = filter_value
    return logits