
# Differential Jet Mass Analysis for Z+Jets Events

This script (`run.py`) performs a differential jet mass analysis for Z+Jets events with NanoAODv9.

## Requirements

This script is supposed be run on LPC. Refer to the following steps to setup the environment. 


1. Use the following command to login -  `ssh -L localhost:8883:localhost:8883 USERNAME@cmslpc-el9.fnal.gov`
2. Move to nobackup directory using `cd nobackup`
3. Running these two commands should create a new file called `shell`
    ```
    curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
    bash bootstrap.sh
    ```
4.  Now launch the singularity container using `./shell coffeateam/coffea-dask:0.7.22-py3.10-gf48fa`
5.  Start the jupyter server `jupyter lab --no-browser --port=8883 --ip 127.0.0.1`. Click on the link to open the jupyter lab interface.
6.  To run the script, open a terminal in the jupyter interface. 

## Script Usage

```bash
python run.py [options]
```

### Options

| Option             | Description                                                                                               | Required | Default              |
|--------------------|-----------------------------------------------------------------------------------------------------------|----------|----------------------|
| `-t`, `--test`     | Run in test mode with minimal number of events                                                            | No       | `False`              |
| `-mc`, `--do_gen`  | Run over MC. (PYTHIA or HERWIG)                                                                              | No       | `False`              |
| `--dask`           | Enable Dask, otherwise runs locally recommended when running over full dataset.                           | No       | `False`              |
| `-s`, `--systematic` | Systematic mode: <br> **1** - No systematics <br> **2** - Minimal jet systematics <br> **3** - All systematics | Yes      | `1`                 |
| `--herwig`         | Use Herwig for MC generation (Boolean). Otherwise uses PYTHIA                                                                  | No       | `False`              |
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
## Information about the files
The output is a pickle file and contains a bunch of histogram. For unfolding purposes, access the `Scikit-HEP hist` object by loading the pickle file and then using the key. Examples can be found in `plotsv2.ipynb`.


## Unfolding Instructions

Coming soon...
