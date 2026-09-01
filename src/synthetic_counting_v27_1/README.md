# Synthetic counting v27.1

v27.1 keeps the complete v27/v24.3 model, data, and separator trace unchanged.
It adds one training-only term during tied atomic-number row calibration.  At
every gold Thinking trace and boundary position before the final count, a
full-vocabulary cross entropy penalizes an atomic number row if it competes
with the frozen correct trace token.  Losses are averaged within each example
before the batch mean, so longer traces do not receive more coefficient.

Only the same ten tied atomic-number rows remain trainable.  Non-thinking has
no trace-safety region and therefore receives the identical final-answer
calibration objective used in v27.  Candidate weights 0.1, 0.3, and 1.0 share
v27's selected learning rate; the Thinking validation split selects one weight
before it is applied to the paired Non-thinking checkpoint and test is opened.
