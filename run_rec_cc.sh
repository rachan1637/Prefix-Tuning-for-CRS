#!/usr/bin/env bash
sbatch --nodes=1 --time=6:00:00 --mem=10G --cpus-per-task=1 --gres=gpu:v100l:1 --account=def-ssanner --mail-user=rayhs.chan@mail.utoronto.ca --mail-type=ALL rec.sh
# salloc --nodes=1 --time=1:00:00 --mem=10G --cpus-per-task=1 --gres=gpu:p100:1 --account=def-ssanner