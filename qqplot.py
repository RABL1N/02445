import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the uploaded Excel file
file_path = '/Users/rasmuslinnemann/Desktop/Courses/Semester2/Project/Group Project/Code/72sample_noise_data.csv'
data = pd.read_csv(file_path)

# Extract relevant columns for ANOVA
anova_data = data[['Noise type', 'BLEU-2']]

# Extract BLEU-2 scores for all noise types
bleu2_scores = anova_data['BLEU-2']

# Generate QQ-plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
sm.qqplot(bleu2_scores, line='s', ax=ax)
ax.set_title('QQ-Plot of BLEU-2 Scores')
ax.set_xlabel('Theoretical Quantiles')
ax.set_ylabel('Sample Quantiles')
ax.grid(True)

# Display the plot
plt.show()
