import os
import sys
import numpy as np
import torch
import argparse
from src.constants import PROJECT_PATH, NUM_TO_LETTER

def test_ribonanza_dimension_fix():
    """
    Test that the RibonanzaNet evaluation fix works correctly with mismatched dimensions.
    This simulates the error that was occurring in the evaluate function.
    """
    try:
        # Import the RibonanzaNet model
        from tools.ribonanzanet.network import RibonanzaNet
        print("Initializing RibonanzaNet model...")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Initialize RibonanzaNet
        ribonanza_net = RibonanzaNet(
            os.path.join(PROJECT_PATH, 'tools/ribonanzanet/config.yaml'),
            os.path.join(PROJECT_PATH, 'tools/ribonanzanet/ribonanzanet.pt'),
            device
        )
        ribonanza_net = ribonanza_net.to(device)
        ribonanza_net.eval()
        
        # Import the evaluation function
        from src.evaluator import self_consistency_score_ribonanzanet
        
        # Create test data with mismatched dimensions
        # Sequence length = 58, mask length = 61 (simulating the error condition)
        sequence_length = 58
        mask_length = 61
        
        # Create dummy sample data
        num_samples = 2
        samples = np.random.randint(0, 4, size=(num_samples, sequence_length))
        
        # Create true sequence
        true_sequence = ''.join([NUM_TO_LETTER[np.random.randint(0, 4)] for _ in range(sequence_length)])
        
        # Create mask that's longer than the sequence (simulating the error condition)
        mask_seq = np.ones(mask_length, dtype=bool)
        
        print(f"Test data created:")
        print(f"  - Sequence length: {sequence_length}")
        print(f"  - Mask length: {mask_length}")
        print(f"  - Samples shape: {samples.shape}")
        print(f"  - True sequence length: {len(true_sequence)}")
        
        # Call the function that previously caused the error
        print("\nTesting RibonanzaNet evaluation with mismatched dimensions...")
        scores = self_consistency_score_ribonanzanet(
            samples, true_sequence, mask_seq, ribonanza_net
        )
        
        print(f"\nSuccess! Function completed without errors.")
        print(f"Returned scores shape: {scores.shape}")
        print(f"Scores: {scores}")
        
        return True
    
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RibonanzaNet dimension mismatch fix")
    args = parser.parse_args()
    
    success = test_ribonanza_dimension_fix()
    
    if success:
        print("\n✅ Test passed! The RibonanzaNet dimension mismatch fix works correctly.")
        sys.exit(0)
    else:
        print("\n❌ Test failed! The fix didn't resolve the dimension mismatch issue.")
        sys.exit(1) 