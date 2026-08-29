# v24: paired no-index trace with count range 1..10

v24 keeps the v22 separator trace and unit loss weights:

```text
<Think> <Sep> marker_1 <Sep> marker_2 ... <Sep> marker_n </Think> <Ans> <n>
```

It changes only the accepted total target-character count from 1..30 to 1..10.
Because this changes the task distribution, v24 retrains both `rope/nonthinking`
and `rope/thinking` from seed 1234. The architecture, 256-character context,
three-character query, natural count sampling, optimizer, 10,000-step schedule,
atomic answer tokens, snapshot cadence, and loss weights remain matched to v22.

The primary diagnostic is not overall accuracy alone. Running-count and final-
count NCC are computed separately for both modes using discovery-selected layers
and a disjoint confirmation split. If Thinking running-count NCC rises strongly
while final-count NCC remains high, the v22 result was plausibly capacity/class-
resolution limited. If running-count NCC remains weak, separator traces likely
use a contextual or distributed progress representation rather than compact
count clusters.

Run the paired experiment:

```bash
python -m synthetic_counting_v24.run_v24 \
  --preset main \
  --stage prepare,train,phase,plots \
  --skip-completed
```
