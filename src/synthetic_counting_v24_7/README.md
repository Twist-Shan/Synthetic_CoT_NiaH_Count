# Synthetic counting v24.7

V24.7 is an answer-query representation-compression control for v24.6. It
keeps the paired models, untied LM head, data, 20-set maximum-entropy sampler,
component-normalized language loss, optimizer, seed, count support, trace
grammar, and 10,000-step schedule fixed. The only new term is a weight-0.1
supervised contrastive loss at the native `<Ans>` query during the task-output
phase. It pulls equal-count query residuals together and separates different
counts. There is no auxiliary decoder and inference is unchanged: success is
judged only from the model's raw autoregressive answer token.
