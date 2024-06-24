import pandas as pd
import scipy.stats as stats

# Load the data
data = pd.read_excel('/Users/rasmuslinnemann/Desktop/Courses/Semester2/Project/Group Project/Code/Noise_combinations.xlsx')

# Ensure 'Noise type(s)' is treated as a string
data['Noise type(s)'] = data['Noise type(s)'].astype(str)

# Clean the data by removing rows with NaN values in 'BLEU-2'
cleaned_data = data.dropna(subset=['BLEU-2'])

# Extract the BLEU-2 scores for the control group
control_group = cleaned_data[cleaned_data['Noise type(s)'] == '0']['BLEU-2']

# Extract the BLEU-2 scores for each noise type (excluding the control group)
noise_groups = cleaned_data[cleaned_data['Noise type(s)'] != '0'].groupby('Noise type(s)')['BLEU-2'].apply(list)

# Prepare the data for ANOVA
grouped_bleu_2_scores = [control_group] + [scores for scores in noise_groups]

# Perform one-way ANOVA
f_value, p_value = stats.f_oneway(*grouped_bleu_2_scores)

# Print the results
print(f"F-value: {f_value}")
print(f"P-value: {p_value}")
