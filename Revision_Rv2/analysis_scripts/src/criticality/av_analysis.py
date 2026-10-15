import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import powerlaw
import pickle
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import os 
from itertools import cycle
from scipy.stats import linregress
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']

# Make mathtext look like Arial
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Arial:italic'
mpl.rcParams['mathtext.bf'] = 'Arial:bold' 

# (optional) sizes + minus sign
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['axes.unicode_minus'] = False

# Global sizes (tweak to taste)
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams["axes.linewidth"] = 1.5       # spine width
mpl.rcParams["xtick.major.width"] =1.5
mpl.rcParams["ytick.major.width"] = 1.5
mpl.rcParams["xtick.minor.width"] = 1.5
mpl.rcParams["ytick.minor.width"] = 1.5
mpl.rcParams["axes.linewidth"] = 1.5       # spine width
mpl.rcParams["xtick.major.width"] =1.5
mpl.rcParams["ytick.major.width"] = 1.5
mpl.rcParams["xtick.minor.width"] = 1.5
mpl.rcParams["ytick.minor.width"] = 1.5
def coarse_grain_avalanche_detection_all_units_intervals(spike_times_all_units, bin_size, intervals, percentage=30):
    """
    Detect avalanches using coarse-grained binning with a custom activity threshold,
    summing spike counts across all units in each time bin, but only within the specified intervals.
    
    Parameters:
    - spike_times_all_units: list of array-like, each entry is a sorted array of spike times (in seconds) for a single unit.
    - bin_size: float, size of the time bin for coarse-graining (in seconds).
    - intervals: list of tuples (start, end) in seconds representing the intervals in which to compute avalanches.
    
    Returns:
    - avalanche_sizes: list of int, sizes of detected avalanches (total spikes across all units).
    - avalanche_durations: list of int, durations of detected avalanches (in bins).
    - ISI: float, average spike rate across all units (spikes per second).
    - avalanche_timestamps: list of float, center timestamps of detected avalanches (in seconds).
    """
    # Flatten spike times to get the overall recording duration
    all_spike_times = np.concatenate(spike_times_all_units)
    max_time = np.max(all_spike_times)
    num_bins = int(np.ceil(max_time / bin_size))
    
    # Initialize binned spike counts for the entire recording
    binned_spikes = np.zeros(num_bins)
    for unit_spikes in spike_times_all_units:
        for spike in unit_spikes:
            bin_idx = int(spike // bin_size)
            # make sure the index is within range
            if bin_idx < num_bins:
                binned_spikes[bin_idx] += 1

    # Compute recording duration based on each unit and calculate an average spike rate (ISI)
    recording_duration = max(np.max(times) for times in spike_times_all_units)
    spike_rates = [len(times) / recording_duration for times in spike_times_all_units]
    ISI = sum(spike_rates) / len(spike_rates)
    
    avalanche_sizes = []
    avalanche_durations = []
    avalanche_timestamps = []
    
    # Process each interval individually
    for interval in intervals:
        start_time, end_time = interval
        # Determine bin indices corresponding to the interval boundaries
        start_bin = int(start_time // bin_size)
        end_bin = int(np.ceil(end_time / bin_size))
        # Clamp indices within valid range
        start_bin = max(0, start_bin)
        end_bin = min(num_bins, end_bin)
        
        # Extract the binned spikes corresponding to the current interval
        interval_bins = binned_spikes[start_bin:end_bin]
        if len(interval_bins) == 0:
            continue
        
        # Define the activity threshold as the 30th percentile of activity in this interval
        activity_threshold = np.percentile(interval_bins, percentage)
        
        current_size = 0
        current_duration = 0
        current_start_bin = None  # absolute bin index of avalanche start
        
        for i, count in enumerate(interval_bins):
            if count >= activity_threshold:
                if current_size == 0:
                    current_start_bin = start_bin + i  # Mark absolute start of avalanche
                current_size += count
                current_duration += 1
            elif current_size > 0:
                # Avalanche has ended, record its properties
                center_bin = current_start_bin + current_duration // 2
                start_times = current_start_bin * bin_size
                end_times = (current_start_bin + current_duration) * bin_size
                avalanche_timestamps.append([start_times, end_times])
                avalanche_sizes.append(current_size)
                avalanche_durations.append(current_duration)
                # Reset counters for the next avalanche
                current_size = 0
                current_duration = 0
                current_start_bin = None
        # If the interval ends while an avalanche is still ongoing, record it
        if current_size > 0:
            start_times = current_start_bin * bin_size
            end_times = (current_start_bin + current_duration) * bin_size
            avalanche_timestamps.append([start_times, end_times])
            avalanche_sizes.append(current_size)
            avalanche_durations.append(current_duration)
    
    return avalanche_sizes, avalanche_durations, ISI, avalanche_timestamps



import seaborn as sns
import matplotlib.pyplot as plt

def plot_power_law_ccdf(data_dict, xmins,xmaxs, xlabel, ylabel, title, animal_name,day,animal,sleep_idx,save_path=None,  csv_path=None, brain_region=None,save_path_svg=None):
    """
    Plot multiple power-law distributions for avalanche sizes or durations using CCDF,
    including fitted power-law curves using the `powerlaw` package.

    Parameters:
    - data_dict: dict, keys are labels (e.g., "NREM", "Wake"), values are lists of avalanche sizes or durations.
    - xlabel: str, label for the x-axis.
    - ylabel: str, label for the y-axis.
    - title: str, title of the plot.
    - save_path: str, optional, path to save the plot.
    """

    if len(xmins) != len(data_dict):
        raise ValueError("The length of xmins must match the number of datasets in data_dict.")

    color_cycle = cycle([sns.color_palette("Set1")[1], sns.color_palette("Set1")[0], "green", "purple", "orange", "brown"])
    fig, ax = plt.subplots(figsize=(8, 6))
    i=0
    alpha_dict = {}
    results = []  # Store results for CSV
    for (label, data), xmin in zip(data_dict.items(), xmins):
        color = next(color_cycle)
        # Fit the power-law distribution with the specified xmin
        
        fit = powerlaw.Fit(data, xmax=xmaxs[i], discrete=True,xmin=xmin)
        
        # Plot the original CCDF
        bins, ccdf = powerlaw.ccdf(data)
        plt.loglog(bins, ccdf, label=f"{label} (α: {fit.alpha:.2f})", color=color,marker='o',markersize=4,linestyle='None')
         
        # Generate theoretical CCDF based on the fit
        xmin = fit.xmin
        alpha = fit.alpha
        # Perform distribution comparisons
        if xmaxs[i]==0:
            theoretical_bins = np.linspace(xmin, max(data), 1000)
        else:
            theoretical_bins = np.linspace(xmin, xmaxs[i], 1000)
        theoretical_ccdf = (theoretical_bins / xmin) ** (-alpha + 1)

        # Scale theoretical CCDF to match empirical data at xmin
        empirical_ccdf_at_xmin = ccdf[np.searchsorted(bins, xmin)]
        theoretical_ccdf *= empirical_ccdf_at_xmin / theoretical_ccdf[0]

        # Plot the theoretical CCDF, starting from xmin
        plt.loglog(theoretical_bins, theoretical_ccdf, label=None, color=color, linestyle='--')

        # Print the KS statistic for this dataset
        ks_stat = fit.power_law.KS()
        print(f"{label} - KS Statistic: {ks_stat:.4f}, Alpha: {fit.alpha:.2f}, Xmin: {fit.xmin},Xmax: {fit.xmax}")
        R1, p_log = fit.distribution_compare('power_law', 'lognormal')
        print(f"Lognormal vs Power-law: R={R1}, p={p_log}")

        R2, p_exp = fit.distribution_compare('power_law', 'exponential')
        print(f"Exponential vs Power-law: R={R2}, p={p_exp}")
              # Add results to the list
        results.append({
            "animal_name": animal_name,
            "day": day,
            "label": label,
            "sleep_idx":sleep_idx,
            "ks_stat": ks_stat,
            "p_log": p_log,
            "p_exp": p_exp,
            "alpha": fit.alpha,
            "xmin": fit.xmin,
            'Region':brain_region,
            "xmax":fit.xmax,
            "animal":animal,
        })
        alpha_dict[label] = fit.alpha
        i=i+1

    # Add labels, title, legend, and grid
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if save_path_svg:
        plt.savefig(save_path_svg, dpi=300, bbox_inches="tight")
    plt.show()
    if csv_path:
        results_df = pd.DataFrame(results)
        if os.path.exists(csv_path):
            # Load existing data
            existing_df = pd.read_csv(csv_path)

            # Remove existing entries with matching animal_name, day, and label
            existing_df = existing_df[~(
                (existing_df["animal_name"] == animal_name) &
                (existing_df["day"] == day) &
                (existing_df["label"].isin(results_df["label"]))&
                (existing_df["sleep_idx"].isin(results_df["sleep_idx"]))&
                (existing_df["Region"].isin(results_df["Region"]))
            )]

            # Append new results
            updated_df = pd.concat([existing_df, results_df], ignore_index=True)
        else:
            # No existing file, create a new DataFrame
            updated_df = results_df

        # Save updated DataFrame back to CSV
        updated_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    return alpha_dict

def plot_dcc(data_dict,xlabel, ylabel, title, animal_name,day,animal,sleep_idx,save_path=None,csv_path=None,brain_region=None,save_path_svg=None):
    """
    Plot multiple power-law distributions for avalanche sizes or durations.

    Parameters:
    - data_dict: dict, keys are labels (e.g., "NREM", "Wake"), values are lists of avalanche sizes or durations.
    - xlabel: str, label for the x-axis.
    - ylabel: str, label for the y-axis.
    - title: str, title of the plot.
    """
    # Generate unique colors for each dataset
    color_cycle = cycle([sns.color_palette("Set1")[1], sns.color_palette("Set1")[0], "green", "purple", "orange", "brown"])
    results = []
    fig, ax = plt.subplots(figsize=(8, 6))
    beta_dict = {} 
    for label, (sizes, durations) in data_dict.items():
        # Ensure data are numpy arrays
        durations = np.array(durations)
        sizes = np.array(sizes)

        valid_indices = (sizes > 0) & (durations > 0)
        sizes = sizes[valid_indices]
        durations = durations[valid_indices]

        # Perform log-log regression to compute beta
        log_durations = np.log(durations)
        log_sizes = np.log(sizes)
        slope, intercept, r_value, p_value, std_err = linregress(log_durations, log_sizes)
        results.append({
            "animal_name": animal_name,
            "day": day,
            "sleep_idx":sleep_idx,
            "label": label,
            "beta": slope,
            "R^2": r_value**2,
            'Region':brain_region,
            "animal":animal
        })
        beta_dict[label] = slope
        print(f"{label}: Beta={slope:.4f}, R^2={r_value**2:.4f}")
        # Plot the data for this dataset
        color = next(color_cycle)
        plt.loglog(durations, sizes, 'o', markersize=4, label=label, alpha=0.7, color=color)
    
    # Add labels, title, and legend
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if save_path_svg:
        plt.savefig(save_path_svg, dpi=300, bbox_inches="tight")
    plt.show()
    if csv_path:
        results_df = pd.DataFrame(results)
        if os.path.exists(csv_path):
            # Load existing data
            existing_df = pd.read_csv(csv_path)

            # Remove existing entries with matching animal_name, day, and label
            existing_df = existing_df[~(
                (existing_df["animal_name"] == animal_name) &
                (existing_df["day"] == day) &
                (existing_df["label"].isin(results_df["label"]))&
                (existing_df["sleep_idx"].isin(results_df["sleep_idx"]))&
                (existing_df["Region"].isin(results_df["Region"]))
            )]

            # Append new results
            updated_df = pd.concat([existing_df, results_df], ignore_index=True)
        else:
            # No existing file, create a new DataFrame
            updated_df = results_df

        # Save updated DataFrame back to CSV
        updated_df.to_csv(csv_path, index=False)
    return beta_dict
def compute_average_size(sizes, durations):
    # Create a DataFrame from sizes and durations
    df = pd.DataFrame({'Size': sizes, 'Duration': durations})
    # Group by 'Duration' and compute the mean 'Size' for each duration
    avg_size_per_duration = df.groupby('Duration')['Size'].mean().reset_index()
    return avg_size_per_duration
def compute_scaling_with_avg(data_dict, xlabel, ylabel, title, animal_name,day,sleep_idx,animal,save_path=None,csv_path=None,brain_region=None,save_path_svg=None):
    """
    Compute avalanche size-duration scaling using CCDF.

    Parameters:
    - data_dict: dict, keys are labels (e.g., "NREM", "Wake"), values are tuples of (sizes, durations).
    - xlabel: str, label for the x-axis.
    - ylabel: str, label for the y-axis.
    - title: str, title of the plot.
    - save_path: str, optional, path to save the plot.

    Returns:
    - scaling_results: dict, beta and R^2 for each label.
    """
    color_cycle = cycle([sns.color_palette("Set1")[1], sns.color_palette("Set1")[0], "green", "purple", "orange", "brown"])
    results = []
    beta_avg_dict = {} 

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, (sizes, durations) in data_dict.items():
        sizes = np.array(sizes)
        durations = np.array(durations)

        # Filter out invalid values
     
        avg_data = compute_average_size(sizes, durations)
        # Perform log-log regression to compute beta
        log_durations = np.log(avg_data['Duration'])
        log_sizes= np.log(avg_data['Size'])
        slope, intercept, r_value, p_value, std_err = linregress(log_durations, log_sizes)

        results.append({
            "animal_name": animal_name,
            "day": day,
            "sleep_idx":sleep_idx,
            "label": label,
            "beta": slope,
            "R^2": r_value**2,
            'Region':brain_region,
            "animal":animal
        })
        # Plot CCDFs
        color = next(color_cycle)
        beta_avg_dict[label]=slope
        plt.loglog(avg_data['Duration'], avg_data['Size'], 'o', markersize=4, label=f"{label}", color=color)
        # Generate smooth x values across the range of your data
        x_fit = np.linspace(avg_data['Duration'].min(), avg_data['Duration'].max(), 200)
        y_fit = np.exp(intercept) * (x_fit ** slope)

        # Plot fit on same log-log axes
        ax.loglog(x_fit, y_fit, '-', color=color, linestyle='--')
        print(f"{label}: Beta={slope:.4f}, R^2={r_value**2:.4f}")
    # Add labels, title, and legend
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if save_path_svg:
        plt.savefig(save_path_svg, dpi=300, bbox_inches="tight")
    plt.show()
    
    if csv_path:
        results_df = pd.DataFrame(results)
        if os.path.exists(csv_path):
            # Load existing data
            existing_df = pd.read_csv(csv_path)

            # Remove existing entries with matching animal_name, day, and label
            existing_df = existing_df[~(
                (existing_df["animal_name"] == animal_name) &
                (existing_df["day"] == day) &
                (existing_df["label"].isin(results_df["label"]))&
                (existing_df["Region"].isin(results_df["Region"]))
            )]

            # Append new results
            updated_df = pd.concat([existing_df, results_df], ignore_index=True)
        else:
            # No existing file, create a new DataFrame
            updated_df = results_df

        # Save updated DataFrame back to CSV
        updated_df.to_csv(csv_path, index=False)
    return beta_avg_dict

def plot_power_law_pdf(data_dict, xmins,xmaxs, xlabel, ylabel, title, animal_name,day,animal,sleep_idx,save_path=None,  csv_path=None, brain_region=None):
    """
    Plot multiple power-law distributions for avalanche sizes or durations using CCDF,
    including fitted power-law curves using the `powerlaw` package.

    Parameters:
    - data_dict: dict, keys are labels (e.g., "NREM", "Wake"), values are lists of avalanche sizes or durations.
    - xlabel: str, label for the x-axis.
    - ylabel: str, label for the y-axis.
    - title: str, title of the plot.
    - save_path: str, optional, path to save the plot.
    """

    if len(xmins) != len(data_dict):
        raise ValueError("The length of xmins must match the number of datasets in data_dict.")

    color_cycle = cycle(["blue", "red", "green", "purple", "orange", "brown"])
    plt.figure(figsize=(8, 6))
    i=0
    alpha_dict = {}
    results = []  # Store results for CSV
    for (label, data), xmin in zip(data_dict.items(), xmins):
        color = next(color_cycle)
        # Fit the power-law distribution with the specified xmin
        
        fit = powerlaw.Fit(data, xmax=xmaxs[i], discrete=True,xmin=xmin)
        
        # Plot the original CCDF
        bins, pdf = powerlaw.pdf(data)
        plt.loglog(bins, pdf, label=f"Original Data [{label}]", color=color,marker='o',markersize=2,linestyle='None')

        # Generate theoretical CCDF based on the fit
        xmin = fit.xmin
        alpha = fit.alpha
        
        # Create theoretical bins starting from xmin
        if xmaxs[i]==0:
            theoretical_bins = np.linspace(xmin, max(data), 1000)
        else:
            theoretical_bins = np.linspace(xmin, xmaxs[i], 1000)
        # Calculate the normalization constant
        C = (alpha - 1) * xmin ** (alpha - 1)

        # Correct formula for theoretical PDF
        theoretical_pdf = C * (theoretical_bins ** -alpha)

        # Scale theoretical CCDF to match empirical data at xmin
        empirical_pdf_at_xmin = pdf[np.searchsorted(bins, xmin)]
        theoretical_pdf *= empirical_pdf_at_xmin / theoretical_pdf[0]

        # Plot the theoretical CCDF, starting from xmin
        plt.loglog(theoretical_bins, theoretical_pdf, label=f"Fitted[{label}]", color=color, linestyle='--')

        # Print the KS statistic for this dataset
        ks_stat = fit.power_law.KS()
        print(f"{label} - KS Statistic: {ks_stat:.4f}, Alpha: {fit.alpha:.2f}, Xmin: {fit.xmin},Xmax: {fit.xmax}")
              # Add results to the list
        results.append({
            "animal_name": animal_name,
            "day": day,
            "label": label,
            "sleep_idx":sleep_idx,
            "ks_stat": ks_stat,
            "alpha": fit.alpha,
            "xmin": fit.xmin,
            'Region':brain_region,
            "xmax":fit.xmax,
            "animal":animal,
        })
        alpha_dict[label] = fit.alpha
        i=i+1

    # Add labels, title, legend, and grid
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    if csv_path:
        results_df = pd.DataFrame(results)
        if os.path.exists(csv_path):
            # Load existing data
            existing_df = pd.read_csv(csv_path)

            # Remove existing entries with matching animal_name, day, and label
            existing_df = existing_df[~(
                (existing_df["animal_name"] == animal_name) &
                (existing_df["day"] == day) &
                (existing_df["label"].isin(results_df["label"]))&
                (existing_df["sleep_idx"].isin(results_df["sleep_idx"]))&
                (existing_df["Region"].isin(results_df["Region"]))
            )]

            # Append new results
            updated_df = pd.concat([existing_df, results_df], ignore_index=True)
        else:
            # No existing file, create a new DataFrame
            updated_df = results_df

        # Save updated DataFrame back to CSV
        updated_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    return alpha_dict