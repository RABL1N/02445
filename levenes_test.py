import pandas as pd
from scipy.stats import levene

# Load the uploaded Excel file
file_path = '/Users/rasmuslinnemann/Desktop/Courses/Semester2/Project/Group Project/Code/72sample_noise_data.csv'
data = pd.read_csv(file_path)

# Extract relevant columns for ANOVA
anova_data = data[['Noise type', 'BLEU-2']]

# Extract BLEU-2 scores for each noise type
bleu2_scores_by_type = [anova_data[anova_data['Noise type'] == i]['BLEU-2'] for i in range(8)]

# Perform Levene's test to check the variance assumption
levene_result = levene(*bleu2_scores_by_type)

print(f"Levene's test statistic: {levene_result.statistic}")
print(f"Levene's test p-value: {levene_result.pvalue}")

if levene_result.pvalue < 0.05:
    print("Conclusion: There is a significant difference in variances.")
else:
    print("Conclusion: There is no significant difference in variances.")
