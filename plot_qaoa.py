import os
import pickle as pkl
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def plot(no_cars, no_routes, p_max_qaoa, p_max_tqa, no_counter, plot_settings, optimizer):
    custom = plot_settings['custom'] #Custom QAOA results
    p_y_min, p_y_max = plot_settings['p_y_scale'] #Probability of success y-scale
    e_y_min, e_y_max = plot_settings['e_y_scale'] #Eigenvalues y-scale
    a_y_min, a_y_max = plot_settings['a_y_scale'] #Approx Quality Ratio y-scale
    no_qbits = no_cars*no_routes
    directory = "results_{}cars{}routes".format(no_cars, no_routes)
    results = {}
    counters = {}
    samples = dict(zip(range(100), [0 for _ in range(100)]))
    for file in os.listdir(directory):
        if ('Ideal_QAOA' not in file and 'Ideal_TQA' not in file) or 'LN' not in file:
            continue
        else:
            try:
                if 'Ideal_QAOA' in file:
                    p_max = int(file[13:15])
                elif 'Ideal_TQA' in file:
                    p_max = int(file[17:19])
            except ValueError:
                continue
            if ('Ideal_QAOA' in file and p_max == p_max_qaoa) or ('Ideal_TQA' in file and p_max == p_max_tqa):
                if 'Ideal_QAOA' in file:
                    sample = int(file[18:21])
                elif 'Ideal_TQA' in file:
                    sample = int(file[22:25])
                else:
                    continue
                key = [sample, p_max]
                if 'Cust' in file:
                    key.append('Cust')
                elif 'Cust' not in file:
                    key.append('Base')
                else:
                    continue
                if 'FOUR' in file:
                    key.append('FOUR')
                else:
                    key.append('NONE')    
                if 'INTP' in file:
                    key.append('INTP')
                else:
                    key.append('none')
                if optimizer=='BOBYQA' and 'BOBYQA' in file:
                    key.append('BOBYQA')
                elif optimizer =='SUBPLEX' and 'SBPLX' in file:
                    key.append('SUBPLEX')
                else:
                    continue
                if 'Ideal_TQA' in file:
                    key.append('TQA')
                elif 'Ideal_QAOA' in file:
                    key.append('norm')
                key = tuple(key)
                with open(directory+"/"+file, "r") as f:
                    b = np.loadtxt(f, delimiter=",")
                    if len(b) != 3*p_max:
                        continue
                    elif key not in results:
                        b = np.asarray(b)
                        b = b.reshape(3,p_max)
                        results[key] = results.get(key, np.zeros(shape=(3,p_max))) + b
                        counters[key] = counters.get(key, 0) + 1
                        samples[sample] += 1
                    else:
                        continue
    good_samples = set({})
    for sample, counter in samples.items():
        if counter == no_counter:
            good_samples.add(sample)
    
    good_results = {}
    for key, result in results.items():
        if key[0] not in good_samples:
            continue
        else:
            p_max = key[1]
            good_results[key[1:len(key)]] = good_results.get(key[1:len(key)], np.zeros(shape=(3,p_max))) + result

    for key, value in good_results.items():
        good_results[key] = value/len(good_samples)

    print("NUMBER OF SAMPLES: {}".format(len(good_samples)))

    #COMPARE METHODS (BASE QAOA)
    plot_keys = {"Probability of Success": 0, "Eigenvalue": 1, "Approximation Quality": 2}
    for data_to_plot in plot_keys:

        plot_key = plot_keys[data_to_plot]

        methods_to_compare = ["FOURIER", "INTERP", "FOURIER AND INTERP", "NAIVE", "TQA"]

        fig = plt.figure(figsize=(12,6), dpi=100)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        #plot methods
        def plot_stuff(optimizer):
            for key, result in good_results.items():
                if custom:
                    if 'Cust' not in key:
                        continue
                else:
                    if 'Cust' in key:
                        continue
                p_max = key[0]
                x = list(range(1, p_max+1))
                for method in methods_to_compare:
                    if method == "FOURIER" and 'FOUR' in key and 'INTP' not in key and 'Cust' not in key and optimizer in key and 'TQA' not in key:
                        data = result[plot_key]
                        method = "FOURIER"
                        index = 0
                        color = colors[index]
                    elif method == "INTERP" and 'FOUR' not in key and 'INTP' in key and 'Cust' not in key and optimizer in key and 'TQA' not in key:
                        data = result[plot_key]
                        method = "INTERP"
                        index = 1
                        color = colors[index]

                    elif method == "FOURIER AND INTERP" and 'FOUR' in key and 'INTP' in key and 'Cust' not in key and optimizer in key and 'TQA' not in key:
                        data = result[plot_key]
                        method = "FOURIER AND INTERP"
                        index = 2
                        color = colors[index]

                    elif method == "FOURIER" and 'FOUR' in key and 'INTP' not in key and 'Cust' in key and optimizer in key and 'TQA' not in key:
                        data = result[plot_key]
                        method = "CUST, FOURIER"
                        index = 3
                        color = colors[index]

                    elif method == "INTERP" and 'FOUR' not in key and 'INTP' in key and 'Cust' in key and optimizer in key and 'TQA' not in key:
                        data = result[plot_key]
                        method = "CUST, INTERP"
                        index = 4
                        color = colors[index]

                    elif method == "FOURIER AND INTERP" and 'FOUR' in key and 'INTP' in key and 'Cust' in key and optimizer in key and 'TQA' not in key:
                        data = result[plot_key]
                        method = "CUST, FOURIER AND INTERP"
                        index = 5
                        color = colors[index]

                    elif method == "NAIVE" and 'FOUR' not in key and 'INTP' not in key and 'Cust' not in key and optimizer in key and 'TQA' not in key:
                        data = result[plot_key]
                        method = "NAIVE"
                        index = 6
                        color = colors[index]

                    elif method == "NAIVE" and 'FOUR' not in key and 'INTP' not in key and 'Cust' in key and optimizer in key and 'TQA' not in key:
                        data = result[plot_key]
                        method = "CUST. NAIVE"
                        index = 7
                        color = colors[index]
                    elif method == "TQA" and 'FOUR' not in key and 'INTP' not in key and 'Cust' not in key and optimizer in key and 'TQA' in key:
                        data = result[plot_key]
                        method = "TQA"
                        index = 8
                        color = colors[index]
                    elif method == "TQA" and 'FOUR' not in key and 'INTP' not in key and 'Cust' in key and optimizer in key and 'TQA' in key:
                        data = result[plot_key]
                        method = "TQA, CUST"
                        index = 9
                        color = colors[index]
                    elif method == "TQA" and 'FOUR' in key and 'INTP' not in key and 'Cust' not in key and optimizer in key and 'TQA' in key:
                        data = result[plot_key]
                        method = "TQA, FOURIER"
                        index = 0
                        color = colors[index]
                    elif method == "TQA" and 'FOUR' in key and 'INTP' not in key and 'Cust' in key and optimizer in key and 'TQA' in key:
                        data = result[plot_key]
                        method = "TQA, CUST, FOURIER"
                        index = 1
                        color = colors[index]                
                    else:
                        continue
                    if 'TQA' in key:
                        plt.plot(x, data, "o-", label = method, linewidth = 2.0, color = color, markersize = 5)
                    else:
                        plt.plot(x, data, "X--", label = method, linewidth = 2.0, color = color, markersize = 5)
            plt.xlabel("p")
            plt.ylabel(["probability", "eigenvalue", "approximation quality"][plot_key])
            x1, x2, y1, y2 = plt.axis()
            if data_to_plot == "Probability of Success":
                plt.axis([x1, x2, p_y_min, p_y_max])
            elif data_to_plot == "Eigenvalue":
                plt.axis([x1, x2, e_y_min, e_y_max])
            elif data_to_plot == "Approximation Quality":
                plt.axis([x1, x2, a_y_min, a_y_max])
            fig.patch.set_facecolor('xkcd:white')
            plt.title(data_to_plot + " vs p (Number of qubits = {}, optimizer={})".format(no_qbits, optimizer), fontsize="x-large")

            handles, labels = plt.gca().get_legend_handles_labels()
            to_sort = list(zip(handles, labels))
            sorted_list = list(sorted( to_sort, key=lambda x:x[1]))
            handles, labels = sorted_list[0], sorted_list[1]
            handles, labels = zip(*[ (handle, label) for handle, label in sorted_list ])

            plt.legend(handles, labels, loc=0, fontsize="large", ncol = 2)

        plt.subplot(111)
        optimizer = optimizer
        plot_stuff(optimizer)
        plt.show() 
