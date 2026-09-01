# Synthetic counting v31

v31 is the one-scalar follow-up to v29. It keeps two fully independent models,
the 256-character input, counts 1--10, uniform semantic-count sampler, 100
marker sets, four-layer/four-head model, no-index separator trace, partial
count-only output untying, optimizer, and 10,000-step schedule unchanged.

Only the component-normalized final-count coefficient changes from 4 to 8.
Trace and structure coefficients remain 1 and 0.1. There is no shared model,
auxiliary objective, decoder, calibration phase, test-time update, or trace
change. This preserves independently interpretable training dynamics and head
bank differentiation for Thinking versus Non-thinking.
