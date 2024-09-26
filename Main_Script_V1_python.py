# %%
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel, mannwhitneyu, shapiro, ttest_ind
import tkinter as tk
from tkinter import filedialog as fd

# %%
# Import the data
root = tk.Tk()                     
root.withdraw()
root.attributes("-topmost", True)
root.focus_force()

Well_File_Path = fd.askopenfilename(filetypes=[('Well File', '.csv')])
df = pd.read_csv(Well_File_Path)

# %%
# Deleting empty columns
df = df.dropna(axis=1, how='all')

# deleting the unnamed column
df = df.drop('Unnamed: 4', axis=1)

df.head()

# %% [markdown]
# The highlighted text is a line of Python code that uses the `dropna` method from the pandas library to remove columns from a DataFrame (`df`) that contain only `NaN` (Not a Number) values.
# 
# Explanation:
# `df.dropna(axis=1, how='all')`:
# `df`: The DataFrame from which columns are being removed.
# `dropna`: A method to remove missing values.
# `axis=1`: Specifies that the operation should be performed on columns (if `axis=0`, it would operate on rows).
# `how='all'`: Specifies that only columns where all values are NaN should be removed.

# %%
# Check the data types
print('Before converting the data types:')
print("Data type of 'Interval Start (S)':", type(df['Interval Start (S)'][0]))
print("Data type of 'Interval End (S)':", type(df['Interval End (S)'][0]))
#print("Data type of 'A1_11':", type(df['A1_11'][0]))

# Python is seeing the intervals as strings instead of floats

# %%
# Convert the columns to numeric types
df['Interval Start (S)'] = pd.to_numeric(df['Interval Start (S)'], errors='coerce')
df['Interval End (S)'] = pd.to_numeric(df['Interval End (S)'], errors='coerce')

# Ensure the values are numerical
interval_start = df['Interval Start (S)'].iloc[0]
interval_end = df['Interval End (S)'].iloc[0]

# Check the data types
print("Data type of 'Interval Start (S)':", interval_start, type(interval_start))
print("Data type of 'Interval End (S)':", interval_end, type(interval_end))

# %% [markdown]
# The +1 is added to the `interval_end.max()` to ensure that the last interval end is included in the range of time bins. This is because np.arange generates values in the half-open interval `[start, stop)`, meaning it includes the start value but excludes the stop value. By adding `+1` to `interval_end.max()`, you ensure that the last interval end is included in the time bins.

# %%
# Define the bin width (e.g., 1 s)
bin_width = 1.0

# Columns to process
columns_to_process = df.columns[5:].tolist()

print(columns_to_process)

# Initialize a dictionary to accumulate spike counts for each column
spike_counts = {col: {} for col in columns_to_process}


# %%
# Initialize spike_counts dictionary
spike_counts = {col: {} for col in columns_to_process}

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    interval_start = row['Interval Start (S)']
    interval_end = row['Interval End (S)']
    
    for col in columns_to_process:
        spike_count = row[col]
        
        # Define the time bins for the current interval
        try:
            time_bins = np.arange(interval_start, interval_end + bin_width, bin_width)
        except ValueError as e:
            continue
        
        # Accumulate spike counts in the bins
        for bin_start in time_bins[:-1]:
            if bin_start not in spike_counts[col]:
                spike_counts[col][bin_start] = 0
            spike_counts[col][bin_start] += spike_count

# Convert the dictionaries to sorted lists of bins and counts
sorted_bins = {col: sorted(spike_counts[col].keys()) for col in columns_to_process}
sorted_counts = {col: [spike_counts[col][bin_start] for bin_start in sorted_bins[col]] for col in columns_to_process}

print(sorted_bins)
print(sorted_counts)

# %%
# Determine the number of columns to process
num_columns = len(columns_to_process)

# Calculate the number of rows and columns for the subplot grid
num_cols = math.ceil(math.sqrt(num_columns))  # Number of columns in the grid
num_rows = math.ceil(num_columns / num_cols)  # Calculate the number of rows needed

print("Number of rows:", num_rows)
print("Number of columns:", num_columns)


# %%
# Create a figure and an array of subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5), sharey=True)

# Ensure axs is always an array
if not isinstance(axs, np.ndarray):
    axs = np.array([axs])

# Flatten the array of subplots for easy iteration
axs = axs.flatten()

# Loop through each column to process
for index, col in enumerate(columns_to_process, 0):  # had to make this 0 bc it's usually 1
    # Calculate the row and column index for the subplot
    row_idx = num_rows - 1 - (index % num_rows)  # Reverse the row index
    col_idx = index // num_rows

    # Calculate the subplot index
    subplot_index = row_idx * num_cols + col_idx

    # Create a subplot for each column
    ax = axs[subplot_index]
    ax.hist(sorted_bins[col], bins=len(sorted_bins[col]), weights=sorted_counts[col], edgecolor='blue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Spike Count')
    ax.set_title(f'Spike Count Histogram for {col}')

    # # Add a vertical span (vspan) to highlight a region
    # vspan_start = 1  # Start position of the vertical span
    # vspan_end = 300    # End position of the vertical span
    # ax.axvspan(vspan_start, vspan_end, color='lightgrey', alpha=0.5, zorder = 1)
    
# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Fine-tune the spacing
plt.show()

# %% [markdown]
# # Visualizing differences pre- and post-stimulation

# %% [markdown]
# ## Defining the first and second segment and creating all the Numpy arrays

# %%
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, shapiro

# Define segment durations
first_segment_duration = 300  # Duration for the first segment
last_segment_duration = 200  # Duration for the last segment
bin_width = 1

# Initialize lists to collect data
all_bins_data = []
all_counts_data = []
first_segment_bins_data = []
first_segment_counts_data = []
last_segment_bins_data = []
last_segment_counts_data = []

# Loop through each column and collect data
for col in columns_to_process:
    bins = sorted_bins[col]
    counts = sorted_counts[col]

    # Calculate segment indices
    first_segment_end_idx = np.searchsorted(bins, first_segment_duration)
    last_segment_start_idx = np.searchsorted(bins, bins[-1] - last_segment_duration + bin_width)

    # Extract segment bins and counts
    first_segment_bins = bins[:first_segment_end_idx]
    first_segment_counts = counts[:first_segment_end_idx]
    last_segment_bins = bins[last_segment_start_idx:]
    last_segment_counts = counts[last_segment_start_idx:]

    # Append data to lists
    all_bins_data.append((col, bins))
    all_counts_data.append((col, counts))
    first_segment_bins_data.append((col, first_segment_bins))
    first_segment_counts_data.append((col, first_segment_counts))
    last_segment_bins_data.append((col, last_segment_bins))
    last_segment_counts_data.append((col, last_segment_counts))

# Define the dtype for the structured arrays
bins_dtype = [('column name', 'U50'), ('bins', 'O')]
counts_dtype = [('column name', 'U50'), ('counts', 'O')]

# Convert lists to structured np.array
structured_bins_array = np.array(all_bins_data, dtype=bins_dtype)
structured_counts_array = np.array(all_counts_data, dtype=counts_dtype)
first_segment_bins_np_array = np.array(first_segment_bins_data, dtype=bins_dtype)
first_segment_counts_np_array = np.array(first_segment_counts_data, dtype=counts_dtype)
last_segment_bins_np_array = np.array(last_segment_bins_data, dtype=bins_dtype)
last_segment_counts_np_array = np.array(last_segment_counts_data, dtype=counts_dtype)

print("structured_bins_array:", structured_bins_array)
print("structured_counts_array:", structured_counts_array)
print("first_segment_bins_np_array:", first_segment_bins_np_array)
print("first_segment_counts_np_array:", first_segment_counts_np_array)
print("last_segment_bins_np_array:", last_segment_bins_np_array)
print("last_segment_counts_np_array:", last_segment_counts_np_array)

# %% [markdown]
# ## Getting the total duration for the pre- and post-

# %%
# Initialize dictionaries to store the total duration
first_segment_total_duration = {}
last_segment_total_duration = {}

# Count the number of bin counts in the first segment and save to dictionary
for value, column_name in enumerate(first_segment_bins_np_array['column name']):
    total_bin_counts = len(first_segment_bins_np_array['bins'][value]) # for every value in the column name, count the number of bins
    first_segment_total_duration[column_name] = total_bin_counts
    print(f"Column: {column_name}, First segment bin counts: {total_bin_counts}")

print("")

# Count the number of bin counts in the last segment and save to dictionary
for value, column_name in enumerate(last_segment_bins_np_array['column name']):
    total_bin_counts = len(last_segment_bins_np_array['bins'][value])
    last_segment_total_duration[column_name] = total_bin_counts
    print(f"Column: {column_name}, Last segment bin counts: {total_bin_counts}")

print("")

# Print the dictionaries
print("First Segment Total Duration:", first_segment_total_duration)
print("Last Segment Total Duration:", last_segment_total_duration)

# %% [markdown]
# ## Getting the total spike counts for the pre- and post-

# %%
# Initialize dictionaries to store the total spike counts
first_segment_total_spike_counts = {}
last_segment_total_spike_counts = {}

# Count the number of spike counts in the first segment and save to dictionary
for value, column_name in enumerate(first_segment_counts_np_array['column name']):
    total_spike_counts = sum(first_segment_counts_np_array['counts'][value]) # for every value in the column name, add the number of spike counts
    first_segment_total_spike_counts[column_name] = total_spike_counts
    print(f"Column: {column_name}, First segment spike counts: {total_spike_counts}")

print("")

# Count the number of spike counts in the last segment and save to dictionary
for value, column_name in enumerate(last_segment_counts_np_array['column name']):
    total_spike_counts = sum(last_segment_counts_np_array['counts'][value])
    last_segment_total_spike_counts[column_name] = total_spike_counts
    print(f"Column: {column_name}, Last segment spike counts: {total_spike_counts}")

print("")

# Print the dictionaries
print("First Segment Total Spike Counts:", first_segment_total_spike_counts)
print("Last Segment Total Spike Counts:", last_segment_total_spike_counts)

# %% [markdown]
# ## Getting the mean firing rates for the pre- and post-

# %%
# Initialize dictionaries to store the mean firing rates
first_segment_mean_firing_rate = {}
last_segment_mean_firing_rate = {}

# Calculate the mean firing rate for the first segment
for column in first_segment_total_spike_counts:
    total_spike_counts = first_segment_total_spike_counts[column]
    total_duration = first_segment_total_duration[column]
    mean_firing_rate = total_spike_counts / total_duration
    first_segment_mean_firing_rate[column] = mean_firing_rate
    print(f"Column: {column}, First segment mean firing rate: {mean_firing_rate}")

print("")

# Calculate the mean firing rate for the last segment
for column in last_segment_total_spike_counts:
    total_spike_counts = last_segment_total_spike_counts[column]
    total_duration = last_segment_total_duration[column]
    mean_firing_rate = total_spike_counts / total_duration
    last_segment_mean_firing_rate[column] = mean_firing_rate
    print(f"Column: {column}, Last segment mean firing rate: {mean_firing_rate}")

print("")

# Print the dictionaries
print("First Segment Mean Firing Rate:", first_segment_mean_firing_rate)
print("Last Segment Mean Firing Rate:", last_segment_mean_firing_rate)

# %% [markdown]
# ## Testing normality and performing t-test or Mann-Whitney U test

# %%
# Perform statistical tests and extract p-values
test_results = []
p_values = {}
for i, col in enumerate(columns_to_process):
    first_segment_counts = first_segment_counts_np_array[i][1]
    last_segment_counts = last_segment_counts_np_array[i][1]
    
    # Perform Shapiro-Wilk test for normality
    first_normal = shapiro(first_segment_counts).pvalue > 0.05
    last_normal = shapiro(last_segment_counts).pvalue > 0.05
    
    if first_normal and last_normal:
        # Perform t-test if both segments are normally distributed
        t_stat, p_value = ttest_ind(first_segment_counts, last_segment_counts)
        test_type = 't-test'
    else:
        # Perform Mann-Whitney U test if either segment is not normally distributed
        t_stat, p_value = mannwhitneyu(first_segment_counts, last_segment_counts)
        test_type = 'Mann-Whitney U test'
    
    # Store results
    test_results.append((col, test_type, t_stat, round(p_value, 3)))
    p_values[col] = p_value

# Print test results
print("Test results for each column:")
for col, test_type, t_stat, p_value in test_results:
    print(f"Column: {col}, Test: {test_type}, Statistic: {t_stat}, P-value: {p_value}")
    

# %% [markdown]
# ## Making the difference plot

# %% [markdown]
# #### Getting the differences Numpy array for the heat map 

# %%
significance_threshold = 0.05  # Define your significance threshold

# Calculate the differences
differences = {column: first_segment_mean_firing_rate[column] - last_segment_mean_firing_rate[column]
               for column in first_segment_mean_firing_rate}

# Filter the columns and differences based on the significance threshold
significant_columns = [col for col in differences if p_values[col] < significance_threshold]
significant_differences = {col: differences[col] for col in significant_columns}
significant_p_values = {col: p_values[col] for col in significant_columns}

# Convert the differences to a numpy array for heatmap
columns = list(differences.keys())
differences_array = np.array(list(differences.values()))

# %% [markdown]
# #### Getting the shape of the difference plot to line up with the MEA grid

# %%
# Determine the size of the reshaped array
num_columns = len(columns)
side_length = int(np.ceil(np.sqrt(num_columns)))

# Pad the arrays to make them square if necessary
pad_size = side_length**2 - num_columns
differences_array_padded = np.pad(differences_array, (0, pad_size), mode='constant', constant_values=0)
columns_array_padded = np.pad(columns, (0, pad_size), mode='constant', constant_values='')
p_values_array_padded = np.pad([p_values.get(col, np.nan) for col in columns], (0, pad_size), mode='constant', constant_values=np.nan)

# Reshape the arrays to the determined size
difference_matrix = differences_array_padded.reshape(side_length, side_length)
columns_matrix = columns_array_padded.reshape(side_length, side_length)
p_values_matrix = p_values_array_padded.reshape(side_length, side_length)

# Transpose the matrices to count upwards
difference_matrix = difference_matrix.T
columns_matrix = columns_matrix.T
p_values_matrix = p_values_matrix.T

# Flip the matrices vertically to start from the bottom-left corner
difference_matrix = np.flipud(difference_matrix)
columns_matrix = np.flipud(columns_matrix)
p_values_matrix = np.flipud(p_values_matrix)

# %% [markdown]
# #### Filtering the heat map to only include the differences and the p values that are significant pre- post- stimulation

# %%
# Create an annotation array with column names, differences, and p-values for significant columns only
annot_array = np.array([
    [
        f'{columns_matrix[i, j]}\n{difference_matrix[i, j]:.2f}\np={p_values_matrix[i, j]:.3f}' if p_values_matrix[i, j] < significance_threshold else f'{columns_matrix[i, j]}'
        for j in range(side_length)
    ]
    for i in range(side_length)
])

# %% [markdown]
# #### Generating the difference plot heat map

# %%
# Create a heat map with a color bar
plt.figure(figsize=(15, 10))  # Adjust the size to fit the grid
ax = sns.heatmap(difference_matrix, cmap='coolwarm', annot=annot_array, fmt='', xticklabels=False, yticklabels=False, cbar=True)
ax.set_title('Difference Plot of Mean Firing Rates')

# Customize the color bar
cbar = ax.collections[0].colorbar
cbar.set_label('Difference in Mean Firing Rates')

plt.show()


