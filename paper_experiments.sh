#!/bin/bash

trap "exit" INT
set -e

current_date_time=`date "+%Y-%m-%d %H:%M:%S"`
echo "Reproducing experiments of the paper Proximal Backpropagation at $current_date_time"
echo "Checking your python version..."
ret=`python -c 'import sys; print(sys.version_info >= (3,5))'`
if [ $ret ]; then
    echo "Python version is >= 3.5"
else 
    echo "We require python version >= 3.5. Note we assume that python3 is aliased by python."
fi

echo "Checking your pdflatex installation..."
command -v pdflatex >/dev/null && echo "Found pdflatex." || { echo >&2 "We need pdflatex to render the results, but it seems like it's not installed.  Aborting."; exit 1; }

package=pgfplots.sty
kpsewhich $package >/dev/null && echo "Found $package" || { echo >&2 "We need the latex package $package to render the results, but it seems like it's not installed.  Aborting."; exit 1; }

package=inputenc.sty
kpsewhich $package >/dev/null && echo "Found $package" || { echo >&2 "We need the latex package $package to render the results, but it seems like it's not installed.  Aborting."; exit 1; }

package=graphicx.sty
kpsewhich $package >/dev/null && echo "Found $package" || { echo >&2 "We need the latex package $package to render the results, but it seems like it's not installed.  Aborting."; exit 1; }

echo "Starting experiments on GPU device 0..."
OUT_DIR=paper_experiments
mkdir -p $OUT_DIR

python proxprop_pytorch.py --learning_rate 1 --tau_prox 5e-2 --optimization_mode prox_exact --outfile $OUT_DIR/MLP_NesterovSGD_exact_tauprox5e-2_lr1 --model MLP --optimizer sgd

python proxprop_pytorch.py --learning_rate 5e-2 --num_epochs 1000 --optimization_mode gradient --outfile $OUT_DIR/MLP_NesterovSGD_gradient_lr5e-2 --model MLP --optimizer sgd

python proxprop_pytorch.py --learning_rate 1 --tau_prox 5e-2 --optimization_mode prox_cg3 --outfile $OUT_DIR/MLP_NesterovSGD_cg3_tauprox5e-2_lr1 --model MLP --optimizer sgd

python proxprop_pytorch.py --learning_rate 1 --tau_prox 5e-2 --optimization_mode prox_cg5 --outfile $OUT_DIR/MLP_NesterovSGD_cg5_tauprox5e-2_lr1 --model MLP --optimizer sgd

python proxprop_pytorch.py --learning_rate 1 --tau_prox 5e-2 --optimization_mode prox_cg10 --outfile $OUT_DIR/MLP_NesterovSGD_cg10_tauprox5e-2_lr1 --model MLP --optimizer sgd

python proxprop_pytorch.py --learning_rate 1e-3 --num_epochs 1000 --optimization_mode gradient --outfile $OUT_DIR/ConvNet_Adam_gradient_lr1e-3 --model ConvNet --optimizer adam

python proxprop_pytorch.py --learning_rate 1e-3 --tau_prox 1 --optimization_mode prox_cg3 --outfile $OUT_DIR/ConvNet_Adam_cg3_tauprox1_lr1e-3 --model ConvNet --optimizer adam

python proxprop_pytorch.py --learning_rate 1e-3 --tau_prox 1 --optimization_mode prox_cg10 --outfile $OUT_DIR/ConvNet_Adam_cg10_tauprox1_lr1e-3 --model ConvNet --optimizer adam

echo "Finished running experiments..."

echo "Extracting data..."
python proxprop_plots.py > /dev/null

echo "Compiling plots..."
pdflatex paper_plots.tex > /dev/null

echo "Tidying up..."
rm paper_plots.aux paper_plots.log
rm paper_experiments/*.tex

current_date_time=`date "+%Y-%m-%d %H:%M:%S"`
echo "Finished reproducing paper experiments at $current_date_time. The resulting plots are in paper_plots.pdf."
