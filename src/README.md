## Hybrid GNN–Transformer Model

In this update we introduce a hybrid model that combines a GVP-GNN encoder with a Transformer-based decoder. This change is motivated by the need to capture long-range interactions and complex sequence dependencies that a purely local message-passing decoder might miss.

### Key Points:
- **Long-Range Dependencies:**  
  The new Transformer decoder allows each nucleotide's embedding (after pooling the multiple conformers) to attend to all other positions, capturing global sequence constraints such as complementary base pairing and avoiding repeated motifs.
  
- **Inspiration from Protein Design:**  
  Similar hybrid approaches in protein inverse folding (e.g., GVP-Transformer models like ESM-IF1) have demonstrated significant improvements in accuracy on large datasets. By retaining the rotation-invariant, geometric feature extraction of the GVP encoder and introducing global self-attention, the model becomes more expressive.

- **Design Pipeline:**
  1. **Raw RNA Data & Featurization:**  
     - **Nodes:**  
       - Raw input: 15 scalars & 4 vectors  
       - After featurization: projected to dimensions (e.g., (64, 4)) then processed to (128, 16) by the GVP encoder.
     2. **Initial Embedding:**  
        The node and edge features are processed via LayerNorm and GVP mappings.
     3. **Encoder Layers (GVP):**  
        Multiple GVP layers update the node embeddings while preserving the geometric information.
     4. **Hybrid Decoder:**  
        - **Pooling:** The multi-conformer node features are averaged to yield a per-node embedding in ℝ¹²⁸.
        - **Transformer:** A lightweight Transformer layer (or stack) then processes these embeddings, enabling global self-attention.
        - **Output Projection:** Finally, the Transformer output is projected to predict one of the 4 RNA bases per nucleotide.
        
- **Efficiency:**  
  By limiting the number of Transformer layers and attention heads, the added O(n²) computational cost is kept under control for typical RNA lengths.

This hybrid approach provides a clear separation between local geometric feature extraction (GVP encoder) and global sequence correlation (Transformer decoder), leading to enhanced expressivity and improved sequence recovery.

