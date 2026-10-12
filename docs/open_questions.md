# Open questions for the DreamMachines team

| # | Question | Why it matters | Current assumption |
|---|---|---|---|
| 1 | **Which parts of the data are human recoveries from failed robot rollouts?** The export has `interventions=false` and no intervention column. **Finding (step 1):** the tray gains exactly one actuator per episode, failures included, but failed episodes never show the placement and end mid-motion → after each failure a human completed the task and that segment was **cut** from this export. Can we get the export with interventions (`interventions=true`)? | RECAP forces human-correction frames to `I_t = True`. It also means a failure episode ends at the takeover moment, and the final arm pose is a strong outcome cue. | Recoveries absent; `forced_positive` hook ready in step 6 |
| 2 | How does a **failed** episode end — timeout, operator stop, or right after the failure? Failures are longer (median 749 vs 586 frames, up to 2229). | Determines what V learns near the end of failures and how to choose `C_fail`. | Ends some time after the failure |
| 3 | Preferred `C_fail` and positive fraction `p_pos`? | Both change the labels. | `C_fail = 2229` (T_max), `p_pos = 0.4`, ablations in steps 2/6 |
| 4 | Are there episode **quality ratings** or failure timestamps? (the export options mention `include_qualities` / `include_unrated`) | Would give a quantitative check for step 7. | None available |
| 5 | How will label quality ultimately be judged — inspection, or downstream policy training? | Sets the bar for step 7. | Inspection (V drops before failure, rises on recovery) |
| 6 | Role of `DreamMachines/actuator_unboxing_4h_diverse` (mentioned in the brief)? | Could pre-train the value function on demonstrations (paper's pre-training stage) before the rollouts. | Not used yet |
| 7 | Office machine: disk space, CUDA driver, can jobs run overnight? | Steps 1 and 4. | ≥ 45 GB free, driver ≥ 570 |
| 8 | Is the 50 fps stream the policy's control rate, and what action-chunk length does the policy use? | Choice of lookahead `N` (50 frames = 1 s) and segment length. | 50 Hz, N = 50 |
