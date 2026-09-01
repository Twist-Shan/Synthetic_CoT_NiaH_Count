# Synthetic counting v57

v57 is a one-factor parallel-capacity control on top of v51. It preserves the
two independently initialized modes, batch 128, 256-character permuted
contexts, counts 1--10, maximum-entropy full-support sampling, the unchanged
separator/no-index trace, pure teacher forcing, the 8/8/16 component loss,
four transformer layers, and the fixed 10,000-step endpoint.

The linked architecture factor changes from 4 heads / 256 residual dimensions
/ 1024 MLP dimensions to 6 heads / 384 residual dimensions / 1536 MLP
dimensions. Head dimension remains exactly 64. This expands the parallel head
bank available for retrieval specialization without adding serial depth or
changing any training objective, token, or inference rule.
