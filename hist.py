import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded Excel file
file_path = '/Users/rasmuslinnemann/Desktop/Courses/Semester2/Project/Group Project/Code/72sample_noise_data.csv'
data = pd.read_csv(file_path)

# Extract relevant columns for ANOVA
anova_data = data[['Noise type', 'BLEU-2']]

# Extract BLEU-2 scores for all noise types
bleu2_scores = anova_data['BLEU-2']

# Generate histogram
plt.figure(figsize=(8, 6))
plt.hist(bleu2_scores, bins=20, edgecolor='k', alpha=0.7)
plt.title('Histogram of BLEU-2 Scores')
plt.xlabel('BLEU-2 Score')
plt.ylabel('Frequency')
plt.grid(True)

# Display the plot
plt.show()
