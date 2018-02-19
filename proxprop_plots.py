"""
Generate PGFPlots TeX output plots from raw plotting data input saved as pickle.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib2tikz import save as tikz_save
import pickle
from itertools import accumulate

output_path = 'paper_experiments/'
input_path = 'paper_experiments/'

def load_data(filename, key, input_path=input_path):
    with open(os.path.join(input_path, filename), 'rb') as f:
        data = pickle.load(f)
    if key not in data['training_metrics']:
        raise ValueError('The provided key "{}" is not a key of the data dict.'.format(key))
    else:
        return data['training_metrics'][key]

#global properties of all plots - if not locally specified otherwise
y_axis = 'val_acc'
y_label = 'Validation Accuracy'
label_fontsize = 'footnotesize'
linewidth = 4
axis_parameter_set = {'xticklabel style={font=\\%s}' % label_fontsize, 'yticklabel style={font=\\%s}' % label_fontsize, 'legend style={line width=%s, fill=gray!7}' % (linewidth/2)}

"""
Exact vs. inexact solves in epochs
"""
y_axis_local = 'full_batch_loss'
plt.figure()
outfile = 'exact_vs_inexact_prox_mlp_plot.tex'
exact_data = load_data('MLP_NesterovSGD_exact_tauprox5e-2_lr1.p', y_axis_local)
sgd_data = load_data('MLP_NesterovSGD_gradient_lr5e-2.p', y_axis_local)
cg3_data = load_data('MLP_NesterovSGD_cg3_tauprox5e-2_lr1.p', y_axis_local)
cg5_data = load_data('MLP_NesterovSGD_cg5_tauprox5e-2_lr1.p', y_axis_local)
cg10_data = load_data('MLP_NesterovSGD_cg10_tauprox5e-2_lr1.p', y_axis_local)
plt.title('CIFAR-10, 3072-4000-1000-4000-10 MLP')
x = range(50)
plt.xlabel('Epochs')
plt.ylabel('Full Batch Training Loss')
plt.plot(x, sgd_data[:50], lw=linewidth, label='BackProp')
plt.plot(x, cg3_data[:50], lw=linewidth, label='ProxProp (cg3)')
plt.plot(x, cg5_data[:50], lw=linewidth, label='ProxProp (cg5)')
plt.plot(x, cg10_data[:50], lw=linewidth, label='ProxProp (cg10)')
plt.plot(x, exact_data[:50], lw=linewidth, label='ProxProp (exact)')
plt.legend(frameon=False)
tikz_save(os.path.join(output_path, outfile), extra_axis_parameters=axis_parameter_set)

"""
Exact vs. inexact solves in epochs validation accuracy
"""
y_axis_local = 'val_acc'
plt.figure()
outfile = 'exact_vs_inexact_prox_mlp_plot_val_acc.tex'
exact_data = load_data('MLP_NesterovSGD_exact_tauprox5e-2_lr1.p', y_axis_local)
sgd_data = load_data('MLP_NesterovSGD_gradient_lr5e-2.p', y_axis_local)
cg3_data = load_data('MLP_NesterovSGD_cg3_tauprox5e-2_lr1.p', y_axis_local)
cg5_data = load_data('MLP_NesterovSGD_cg5_tauprox5e-2_lr1.p', y_axis_local)
cg10_data = load_data('MLP_NesterovSGD_cg10_tauprox5e-2_lr1.p', y_axis_local)
plt.title('CIFAR-10, 3072-4000-1000-4000-10 MLP')
x = range(50)
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.plot(x, sgd_data[:50], lw=linewidth, label='BackProp')
plt.plot(x, cg3_data[:50], lw=linewidth, label='ProxProp (cg3)')
plt.plot(x, cg5_data[:50], lw=linewidth, label='ProxProp (cg5)')
plt.plot(x, cg10_data[:50], lw=linewidth, label='ProxProp (cg10)')
plt.plot(x, exact_data[:50], lw=linewidth, label='ProxProp (exact)')
plt.legend(frameon=False)
tikz_save(os.path.join(output_path, outfile), extra_axis_parameters=axis_parameter_set)

"""
Inexact solve vs. SGD in time
"""
y_axis_local = 'full_batch_loss'
plt.figure()
outfile = 'inexact_prox_vs_sgd_mlp_plot.tex'
exact_data = load_data('MLP_NesterovSGD_exact_tauprox5e-2_lr1.p', y_axis_local)
sgd_data = load_data('MLP_NesterovSGD_gradient_lr5e-2.p', y_axis_local)
cg3_data = load_data('MLP_NesterovSGD_cg3_tauprox5e-2_lr1.p', y_axis_local)
cg5_data = load_data('MLP_NesterovSGD_cg5_tauprox5e-2_lr1.p', y_axis_local)
cg10_data = load_data('MLP_NesterovSGD_cg10_tauprox5e-2_lr1.p', y_axis_local)
plt.title('CIFAR-10, 3072-4000-1000-4000-10 MLP')
epoch_time_exact = load_data('MLP_NesterovSGD_exact_tauprox5e-2_lr1.p', 'epoch_time')
epoch_time_sgd = load_data('MLP_NesterovSGD_gradient_lr5e-2.p', 'epoch_time')
epoch_time_cg3 = load_data('MLP_NesterovSGD_cg3_tauprox5e-2_lr1.p', 'epoch_time')
epoch_time_cg5 = load_data('MLP_NesterovSGD_cg5_tauprox5e-2_lr1.p', 'epoch_time')
epoch_time_cg10 = load_data('MLP_NesterovSGD_cg10_tauprox5e-2_lr1.p', 'epoch_time')
x_exact = list(accumulate(epoch_time_exact))
x_cg3 = list(accumulate(epoch_time_cg3))
x_cg5 = list(accumulate(epoch_time_cg5))
x_cg10 = list(accumulate(epoch_time_cg10))
x_sgd = list(accumulate(epoch_time_sgd))
plt.xlabel('Time [s]')
plt.ylabel('Full Batch Training Loss')
x_max = 5 * 60
x_sgd = [x for x in x_sgd if x <= x_max]
x_cg3 = [x for x in x_cg3 if x <= x_max]
x_cg5 = [x for x in x_cg5 if x <= x_max]
x_cg10 = [x for x in x_cg10 if x <= x_max]
x_exact = [x for x in x_exact if x <= x_max]
sgd_data = sgd_data[:len(x_sgd)]
cg3_data = cg3_data[:len(x_cg3)]
cg5_data = cg5_data[:len(x_cg5)]
cg10_data = cg10_data[:len(x_cg10)]
exact_data = exact_data[:len(x_exact)]
plt.plot(x_sgd, sgd_data, lw=linewidth, label='BackProp')
plt.plot(x_cg3, cg3_data, lw=linewidth, label='ProxProp (cg3)')
plt.plot(x_cg5, cg5_data, lw=linewidth, label='ProxProp (cg5)')
plt.plot(x_cg10, cg10_data, lw=linewidth, label='ProxProp (cg10)')
plt.plot(x_exact, exact_data, lw=linewidth, label='ProxProp (exact)')
plt.legend(frameon=False)
tikz_save(os.path.join(output_path, outfile), extra_axis_parameters=axis_parameter_set)

"""
ConvNet: ProxProp vs. SGD directions with Adam full batch loss in epochs
"""
y_axis = 'full_batch_loss'
y_label = 'Full Batch Loss'
plt.figure()
outfile = 'proxprop_vs_sgd_adam_convnet_epochs_plot.tex'
cg3_data = load_data('ConvNet_Adam_cg3_tauprox1_lr1e-3.p', y_axis)
cg10_data = load_data('ConvNet_Adam_cg10_tauprox1_lr1e-3.p', y_axis)
sgd_data = load_data('ConvNet_Adam_gradient_lr1e-3.p', y_axis)
plt.title('CIFAR-10, Convolutional Neural Network')
x = range(len(cg3_data))
plt.xlabel('Epochs')
plt.ylabel(y_label)
plt.plot(x, sgd_data[:51], lw=linewidth, label='Adam + BackProp')
plt.plot(x, cg3_data, lw=linewidth, label='Adam + ProxProp (3 cg)')
plt.plot([], []) # null plot to advance the color cycler
plt.plot(x, cg10_data, lw=linewidth, label='Adam + ProxProp (10 cg)')
plt.legend(frameon=False)
tikz_save(os.path.join(output_path, outfile), extra_axis_parameters=axis_parameter_set)

"""
ConvNet: ProxProp vs. SGD directions with Adam full batch loss in time
"""
plt.figure()
outfile = 'proxprop_vs_sgd_adam_convnet_time_plot.tex'
cg3_data = load_data('ConvNet_Adam_cg3_tauprox1_lr1e-3.p', y_axis)
cg10_data = load_data('ConvNet_Adam_cg10_tauprox1_lr1e-3.p', y_axis)
sgd_data = load_data('ConvNet_Adam_gradient_lr1e-3.p', y_axis)
plt.title('CIFAR-10, Convolutional Neural Network')
epoch_time_cg3 = load_data('ConvNet_Adam_cg3_tauprox1_lr1e-3.p', 'epoch_time')
x_cg3 = list(accumulate(epoch_time_cg3))
max_cg3_time = max(x_cg3)

epoch_time_sgd = load_data('ConvNet_Adam_gradient_lr1e-3.p', 'epoch_time')
x_sgd = list(accumulate(epoch_time_sgd))
x_sgd = [x for x in x_sgd if x <= max_cg3_time]

epoch_time_cg10 = load_data('ConvNet_Adam_cg10_tauprox1_lr1e-3.p', 'epoch_time')
x_cg10 = list(accumulate(epoch_time_cg10))
x_cg10 = [x for x in x_cg10 if x <= max_cg3_time]

sgd_data = sgd_data[:len(x_sgd)]
cg3_data = cg3_data[:len(x_cg3)]
cg10_data = cg10_data[:len(x_cg10)]

plt.xlabel('Time [s]')
plt.ylabel(y_label)
plt.plot(x_sgd, sgd_data, lw=linewidth, label='Adam + BackProp')
plt.plot(x_cg3, cg3_data, lw=linewidth, label='Adam + ProxProp (3 cg)')
plt.plot([], []) # null plot to advance the color cycler
plt.plot(x_cg10, cg10_data, lw=linewidth, label='Adam + ProxProp (10 cg)')
plt.legend(frameon=False)
tikz_save(os.path.join(output_path, outfile), extra_axis_parameters=axis_parameter_set)

"""
ConvNet: ProxProp vs. SGD directions with Adam validation accuracy in epochs
"""
y_axis = 'val_acc'
y_label = 'Validation Accuracy'
plt.figure()
outfile = 'proxprop_vs_sgd_adam_convnet_epochs_plot_val.tex'
cg3_data = load_data('ConvNet_Adam_cg3_tauprox1_lr1e-3.p', y_axis)
cg10_data = load_data('ConvNet_Adam_cg10_tauprox1_lr1e-3.p', y_axis)
sgd_data = load_data('ConvNet_Adam_gradient_lr1e-3.p', y_axis)
plt.title('CIFAR-10, Convolutional Neural Network')
x = range(len(cg3_data))
plt.xlabel('Epochs')
plt.ylabel(y_label)
plt.plot(x, sgd_data[:51], lw=linewidth, label='Adam + BackProp')
plt.plot(x, cg3_data, lw=linewidth, label='Adam + ProxProp (3 cg)')
plt.plot([], []) # null plot to advance the color cycler
plt.plot(x, cg10_data, lw=linewidth, label='Adam + ProxProp (10 cg)')
plt.legend(frameon=False)
tikz_save(os.path.join(output_path, outfile), extra_axis_parameters=axis_parameter_set)

"""
ConvNet: ProxProp vs. SGD directions with Adam validation accuracy in time
"""
y_axis = 'val_acc'
y_label = 'Validation Accuracy'
plt.figure()
outfile = 'proxprop_vs_sgd_adam_convnet_time_plot_val.tex'
cg3_data = load_data('ConvNet_Adam_cg3_tauprox1_lr1e-3.p', y_axis)
cg10_data = load_data('ConvNet_Adam_cg10_tauprox1_lr1e-3.p', y_axis)
sgd_data = load_data('ConvNet_Adam_gradient_lr1e-3.p', y_axis)
plt.title('CIFAR-10, Convolutional Neural Network')
epoch_time_cg3 = load_data('ConvNet_Adam_cg3_tauprox1_lr1e-3.p', 'epoch_time')
x_cg3 = list(accumulate(epoch_time_cg3))
max_cg3_time = max(x_cg3)

epoch_time_sgd = load_data('ConvNet_Adam_gradient_lr1e-3.p', 'epoch_time')
x_sgd = list(accumulate(epoch_time_sgd))
x_sgd = [x for x in x_sgd if x <= max_cg3_time]

epoch_time_cg10 = load_data('ConvNet_Adam_cg10_tauprox1_lr1e-3.p', 'epoch_time')
x_cg10 = list(accumulate(epoch_time_cg10))
x_cg10 = [x for x in x_cg10 if x <= max_cg3_time]

sgd_data = sgd_data[:len(x_sgd)]
cg3_data = cg3_data[:len(x_cg3)]
cg10_data = cg10_data[:len(x_cg10)]

plt.xlabel('Time [s]')
plt.ylabel(y_label)
plt.plot(x_sgd, sgd_data, lw=linewidth, label='Adam + BackProp')
plt.plot(x_cg3, cg3_data, lw=linewidth, label='Adam + ProxProp (3 cg)')
plt.plot([], []) # null plot to advance the color cycler
plt.plot(x_cg10, cg10_data, lw=linewidth, label='Adam + ProxProp (10 cg)')
plt.legend(frameon=False)
tikz_save(os.path.join(output_path, outfile), extra_axis_parameters=axis_parameter_set)
