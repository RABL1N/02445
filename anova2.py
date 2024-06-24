import pandas as pd
import scipy.stats as stats

# Load the uploaded Excel file
file_path = '/Users/rasmuslinnemann/Desktop/Courses/Semester2/Project/Group Project/Code/72sample_noise_data.csv'
data = pd.read_csv(file_path)

# Extract relevant columns for ANOVA
anova_data = data[['Noise type', 'BLEU-2']]

# Separate the control group (Noise type 0) from the rest
control_group = anova_data[anova_data['Noise type'] == 0]['BLEU-2']

# List all noise types including the control group (0 to 7)
noise_types_all = range(8)

# Perform one-way ANOVA comparing each noise type against the control group (Noise type 0)
anova_results_all = {}
for noise_type in noise_types_all:
    if noise_type == 0:
        continue
    noise_group = anova_data[anova_data['Noise type'] == noise_type]['BLEU-2']
    anova_results_all[noise_type] = stats.f_oneway(control_group, noise_group)

# Display the results
for noise_type, result in anova_results_all.items():
    print(f"Noise type {noise_type}:")
    print(f"  F-statistic: {result.statistic}")
    print(f"  p-value: {result.pvalue}")
    if result.pvalue < 0.05:
        print("  Conclusion: Significant difference compared to the control group.")
    else:
        print("  Conclusion: No significant difference compared to the control group.")
