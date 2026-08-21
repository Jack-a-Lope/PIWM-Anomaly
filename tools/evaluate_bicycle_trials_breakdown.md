# `evaluate_bicycle_trials.py` Breakdown

This document describes the current `tools/evaluate_bicycle_trials.py` script. The script evaluates a fixed, hand-tuned bicycle-style dynamics model against DonkeyCar trial logs. It does not currently fit model parameters from the data.

## Purpose

The script loads trial CSV logs, rolls out a simple vehicle model from the first logged state, compares that predicted trajectory against the actual logged trajectory, and writes metrics and optional plots.

The main outputs are:

- `fit_params.json`: the fixed model parameters and run settings used.
- `run_metrics.csv`: one row of metrics per log file.
- `case_summary.csv`: metrics grouped by anomaly case and intensity.
- `plots/rollout_*.png`: actual path vs model rollout, when `matplotlib` is installed.

## Required Input Files

The script searches under `--trials-dir` for files matching:

```text
**/runs/*/log_*.csv
```

Each CSV must include these columns:

```text
timestamp_ms
steering_act
throttle_act
pos_x
pos_z
speed
yaw
anomaly_param
anomaly_intensity
```

Depending on `--control-mode`, the script may also use these optional columns:

```text
steering_delayed
throttle_delayed
steering_cmd
throttle_cmd
```

If the selected optional control column is missing, the script falls back to `steering_act` or `throttle_act`.

## Dependencies

Required:

```text
numpy
```

Optional:

```text
matplotlib
```

If `matplotlib` is unavailable, the script still writes CSV and JSON metrics but skips plot generation.

## Command-Line Arguments

```text
--trials-dir
```

Root directory searched for `log_*.csv` files.

```text
--output-dir
```

Directory where metrics, parameters, and plots are written.

```text
--control-mode {delayed,actual,cmd}
```

Selects which steering and throttle columns are used:

- `delayed`: use `steering_delayed` and `throttle_delayed` if present.
- `actual`: use `steering_act` and `throttle_act`.
- `cmd`: use `steering_cmd` and `throttle_cmd` if present.

```text
--plot-limit
```

Maximum number of rollout plots to save.

```text
--skip-frames
```

Drops the first N valid rows from every log. This is useful when the beginning of each trial contains spawn/controller transients.

```text
--steering-delay-frames
```

Applies an additional frame delay to the selected steering signal. A value of `2` means each frame uses the steering command from two frames earlier.

```text
--max-step-distance
```

If set, splits logs at position jumps larger than this threshold and keeps the longest continuous segment. This helps remove teleport/reset/tracking artifacts.

## Data Loading

### `parse_float(value)`

Converts CSV values to floats. Empty or missing values become `np.nan`.

### `clean_set(value)`

Normalizes anomaly labels. Empty values or `{}` become `nominal`; set-like labels such as `{noise,cam_pitch}` become `noise+cam_pitch`.

### `read_log(path, control_mode, skip_frames=0, max_step_distance=None)`

Reads one CSV into a dictionary of NumPy arrays:

```text
timestamp_ms
x
z
yaw
speed
steer
throttle
dt
path
case
intensity
```

Important processing steps:

- Validates required columns.
- Selects steering/throttle columns based on `--control-mode`.
- Converts `yaw` from degrees to radians.
- Removes rows with non-finite numeric values.
- Optionally skips initial frames.
- Optionally removes discontinuities using `--max-step-distance`.
- Unwraps yaw with `np.unwrap()`.
- Computes `dt` from timestamps in seconds.
- Replaces bad `dt` values with the median valid timestep.

## Log Discovery

### `discover_logs(trials_dir)`

Returns sorted paths matching:

```text
Path(trials_dir).glob("**/runs/*/log_*.csv")
```

## Steering Delay

### `apply_steering_delay(logs, delay_frames)`

Mutates each loaded log by replacing `log["steer"]` with a delayed version of itself.

For example, with `delay_frames = 2`:

```text
new_steer[0] = old_steer[0]
new_steer[1] = old_steer[0]
new_steer[2] = old_steer[0]
new_steer[3] = old_steer[1]
...
```

This is a crude discrete delay model. It is separate from the steering response/lag parameter inside the bicycle model.

## Heading Conventions

### `heading_candidates(yaw)`

Maps yaw into possible world-frame heading vectors. This exists because different simulators/loggers use different conventions for how yaw maps onto `x` and `z`.

Available conventions:

| Name | x heading | z heading |
| --- | --- | --- |
| `cos_sin` | `cos(yaw)` | `sin(yaw)` |
| `cos_neg_sin` | `cos(yaw)` | `-sin(yaw)` |
| `neg_cos_sin` | `-cos(yaw)` | `sin(yaw)` |
| `neg_cos_neg_sin` | `-cos(yaw)` | `-sin(yaw)` |
| `sin_cos` | `sin(yaw)` | `cos(yaw)` |
| `sin_neg_cos` | `sin(yaw)` | `-cos(yaw)` |
| `neg_sin_cos` | `-sin(yaw)` | `cos(yaw)` |
| `neg_sin_neg_cos` | `-sin(yaw)` | `-cos(yaw)` |

The current default is:

```python
"heading_convention": "sin_cos"
```

## Model Parameters

### `default_bicycle_params()`

Returns the fixed parameters used for every run.

Current defaults:

```python
{
    "heading_convention": "sin_cos",
    "x_speed_scale": 0.9,
    "x_bias": 0.0,
    "z_speed_scale": 0.9,
    "z_bias": 0.0,
    "steer_over_wheelbase": 1.395,
    "yaw_bias": 0.0,
    "throttle_accel_gain": 14.7,
    "speed_drag_gain": -0.18,
    "accel_bias": -1.2,
    "steering_staturation": 0.4,
    "steer_response": 0.3,
}
```

Parameter meanings:

| Parameter | Meaning |
| --- | --- |
| `heading_convention` | Chooses how yaw maps to world `x,z` direction. |
| `x_speed_scale` | Scales speed contribution to `x` movement. |
| `z_speed_scale` | Scales speed contribution to `z` movement. |
| `x_bias` | Constant world-frame drift in `x`. |
| `z_bias` | Constant world-frame drift in `z`. |
| `steer_over_wheelbase` | Base yaw-rate gain from speed and steering. |
| `yaw_bias` | Constant yaw drift rate. |
| `throttle_accel_gain` | Converts throttle into acceleration. |
| `speed_drag_gain` | Speed-proportional drag or damping; usually negative. |
| `accel_bias` | Constant acceleration offset. |
| `steering_staturation` | Saturation strength for large steering values. The name is currently misspelled. |
| `steer_response` | First-order steering lag response. `1.0` means instant response; smaller values mean slower response. |

## Dynamics Step

### `step(state, steer, throttle, dt, params)`

Applies one model transition.

Conceptually, the model state is intended to be:

```text
[x, z, yaw, speed, steer_state]
```

The position update is:

```text
x_next = x + x_speed_scale * speed * heading_x * dt + x_bias * dt
z_next = z + z_speed_scale * speed * heading_z * dt + z_bias * dt
```

The steering command is saturated:

```text
effective_steer_cmd = steer / (1 + steering_saturation * abs(steer))
```

The intended steering lag update is:

```text
steer_state_next = steer_state + steer_response * (effective_steer_cmd - steer_state)
```

The yaw update is:

```text
yaw_next = yaw + steer_over_wheelbase * speed * steer_state_next * dt + yaw_bias * dt
```

The speed update is:

```text
speed_next =
    speed
  + (throttle_accel_gain * throttle
     + speed_drag_gain * speed
     + accel_bias) * dt
```

The returned speed is clipped to zero:

```text
speed_next = max(0.0, speed_next)
```

### Current Code Issue

The current `step()` implementation tries to use `steer_state`, but it only unpacks four values:

```python
x, z, yaw, speed = state
```

Then it later uses:

```python
steer_state = steer_state + params["steer_response"] * (effective_steer_cmd - steer_state)
```

That means `steer_state` is referenced before assignment. Since `evaluate_log()` now creates 5-element states, `step()` should unpack five values:

```python
x, z, yaw, speed, steer_state = state
```

This markdown documents the intended 5-state model, but the current Python file needs that unpacking line corrected before the steering-lag version will run.

## Evaluation

### `evaluate_log(log, params)`

Evaluates one log in two ways:

1. **One-step prediction**

   Each prediction starts from the true measured state at frame `i`.

   ```text
   true state at i + logged action at i -> predicted state at i+1
   ```

   This tests local transition accuracy.

2. **Rollout prediction**

   The model starts at the first measured state and then feeds its own prediction into the next step.

   ```text
   predicted state at i + logged action at i -> predicted state at i+1
   ```

   This tests accumulated drift over the whole run.

The rollout state is initialized as:

```python
rollout[0] = [x0, z0, yaw0, speed0, steer0]
```

The one-step measured state is initialized from the logged state at each frame:

```python
[log["x"][i], log["z"][i], log["yaw"][i], log["speed"][i], log["steer"][i]]
```

Only position error is reported. Yaw, speed, and steering-state errors are not currently written as metrics.

## Metrics

The script computes Euclidean position error in the `x,z` plane:

```text
sqrt((pred_x - true_x)^2 + (pred_z - true_z)^2)
```

Per-run metrics:

| Metric | Meaning |
| --- | --- |
| `one_step_pos_rmse` | Local one-frame position RMSE. |
| `one_step_pos_mae` | Mean one-frame position error. |
| `one_step_pos_p50` | Median one-frame position error. |
| `one_step_pos_p95` | 95th percentile one-frame position error. |
| `one_step_pos_max` | Largest one-frame position error. |
| `rollout_pos_rmse` | Full-rollout position RMSE. |
| `rollout_pos_mae` | Mean full-rollout position error. |
| `rollout_pos_p50` | Median full-rollout position error. |
| `rollout_pos_p95` | 95th percentile full-rollout position error. |
| `rollout_final_pos_error` | Distance between final predicted position and final actual position. |
| `frames` | Number of usable frames in the log. |

One-step error can be small while rollout error is large. That means the local transition is close, but yaw, speed, or steering-state errors compound over time.

## Group Summary

### `group_summary(rows)`

Groups run metrics by:

```text
(case, intensity)
```

It averages each metric across runs in the group and sums the frame counts.

Important caveat: grouped RMSE values are simple averages of per-run RMSE values. They are not weighted by frame count.

## CSV Writing

### `write_csv(path, rows, fields)`

Writes selected fields from each row dictionary in a fixed order.

The `rollout` array is intentionally not written to CSV because it is large and only used for plotting.

## Plotting

### `plot_rollouts(output_dir, evaluated, limit)`

Writes comparison plots for the first `limit` evaluated logs.

Each plot contains:

- Blue line: actual logged `x,z` path.
- Orange line: model rollout `x,z` path.

The plot uses:

```python
ax.axis("equal")
```

This keeps `x` and `z` distances visually comparable. If the model rollout explodes or drifts far away, the actual path may look tiny because both paths share one axis scale.

## Main Flow

### `main()`

The script runs this sequence:

1. Parse command-line arguments.
2. Discover all `log_*.csv` files.
3. Read and clean each log.
4. Apply optional steering frame delay.
5. Create the output directory.
6. Load fixed default bicycle parameters.
7. Evaluate every log.
8. Group metrics by case and intensity.
9. Write `fit_params.json`.
10. Write `run_metrics.csv`.
11. Write `case_summary.csv`.
12. Write rollout plots if `matplotlib` is available.
13. Print a short console summary.

## Example Run

PowerShell example:

```powershell
python tools\evaluate_bicycle_trials.py `
  --trials-dir "C:\Users\Jack\physical-donkeycar-anomalies2\phda\physical-donkeycar-anomalies\trials" `
  --output-dir outputs\bicycle_eval `
  --control-mode actual `
  --skip-frames 5 `
  --max-step-distance 1.0
```

## Practical Interpretation

Use the metrics this way:

- If `one_step_pos_rmse` is high, the local dynamics model is wrong or the data has frame/timestamp issues.
- If `one_step_pos_rmse` is low but `rollout_pos_rmse` is high, yaw, speed, or steering-state drift is accumulating.
- If `rollout_final_pos_error` is low but plots look bad, final error is hiding mid-run path error.
- If plots disagree across runs in opposite directions, avoid tuning one scalar globally; inspect timing, speed drift, and actuator lag separately.

## Current Maintenance Notes

- `math` is imported but not used.
- `steering_staturation` is misspelled. The typo is harmless only while every reference uses the same spelling.
- The current `step()` function should unpack `steer_state` from the 5-element state before the steering-response model can run.
- The script still writes `fit_params.json`, but the parameters are currently fixed defaults, not fitted values.
