
# Differential Jet Mass Analysis for Z+Jets Events

This script (`run.py`) performs a differential jet mass analysis for Z+Jets events with NanoAODv9, using various options for MC generation, systematic variations, and optional Dask support.

## Requirements



## Script Usage

```bash
python run.py [options]
```

### Options

| Option             | Description                                                                                               | Required | Default              |
|--------------------|-----------------------------------------------------------------------------------------------------------|----------|----------------------|
| `-t`, `--test`     | Run in test mode with minimal number of events                                                            | No       | `False`              |
| `-mc`, `--do_gen`  | Generate MC data (Boolean).                                                                               | No       | `False`              |
| `--dask`           | Enable Dask, otherwise runs locally recommended when running over full dataset.                           | No       | `False`              |
| `-s`, `--systematic` | Systematic mode: <br> **1** - No systematics <br> **2** - Minimal jet systematics <br> **3** - All systematics | Yes      | N/A                  |
| `--herwig`         | Use Herwig for MC generation (Boolean).                                                                   | No       | `False`              |
| `-o`, `--output`   | Path to save output (Overrides default output naming convention).                                         | No       | Automatically generated based on options |


## Examples

1. **Run in Test Mode with Default Options:**
   ```bash
   python run.py --test -s 1
   ```

2. **Run with MC with Minimal Jet Systematics and Default Output:**
   ```bash
   python run.py -mc -s 2
   ```

3. **Run with Dask Enabled, Using Herwig for MC Generation:**
   ```bash
   python run.py -mc --herwig --dask -s 3
   ```

4. **Custom Output Path with All Systematics:**
   ```bash
   python run.py -s 3 -o custom_output.pkl
   ```

