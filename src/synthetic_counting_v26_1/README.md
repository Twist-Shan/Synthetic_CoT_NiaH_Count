# Synthetic counting v26.1

V26.1 freezes both complete v26 transformer backbones and validation-selects
one shared schedule for updating only the ten existing atomic-number rows of
their native untied LM heads.  It changes neither trace grammar nor inference,
and test prompts are opened only after the schedule is fixed.
