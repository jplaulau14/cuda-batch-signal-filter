# cuda-batch-signal-filter

CUDA batch signal processing project for the CUDA at Scale independent project.

The program processes many CSV signal files with a GPU moving-average filter.
If no input files exist, it generates synthetic noisy signals first, then runs
the CUDA kernel over each file and writes filtered outputs plus a runtime
summary.

## Why This Meets The Assignment

- Uses a custom CUDA kernel for signal processing.
- Processes many small signal arrays in one run.
- Provides a CLI with arguments.
- Includes `Makefile` and `run.sh`.
- Produces proof artifacts in `output/`.

## Build

Run this inside the Coursera CUDA lab or any machine with `nvcc`:

```bash
make clean build
```

## Run

```bash
./run.sh
```

Equivalent manual command:

```bash
./batch_signal_filter --input data/input --output output/filtered --window 7 --limit 100 --generate 100 --length 2048
```

## CLI Arguments

- `--input`: directory containing input CSV signal files
- `--output`: directory for filtered CSV files
- `--window`: moving-average window size
- `--limit`: maximum number of files to process
- `--generate`: number of synthetic inputs to generate if none exist
- `--length`: number of samples per generated signal

## Output Artifacts

After running, the project writes:

- `output/run.log`: terminal log from `run.sh`
- `output/summary.csv`: per-file processing statistics
- `output/filtered/*.csv`: filtered signal outputs

These files are the proof that the CUDA code ran on many signal inputs.

## CUDA Kernel

The kernel assigns one thread to one signal sample. Each thread computes the
average of neighboring samples inside the requested window and writes the
smoothed value to the output array.

## Lessons Learned

This project keeps the GPU work simple but real: copy signal data to device
memory, run a custom CUDA kernel, copy the result back, and repeat across many
files. The main constraint is transfer overhead; small signals are easy to
process, but batching many files makes the GPU work visible in the logs.
