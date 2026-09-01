# Synthetic counting v24.6

V24.6 is a readout-bridge control for v24.5. It keeps the paired models, data,
20-set maximum-entropy sampler, component-normalized loss, optimizer, seed,
count support, trace grammar, and 10,000-step schedule fixed. The only
substantive change is untying the LM output projection from the input token
embedding. The two matrices are copied equal at initialization, so step-zero
logits match v24.5 exactly; they can receive independent gradients thereafter.
