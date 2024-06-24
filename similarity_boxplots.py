import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
data = pd.read_csv('/Users/rasmuslinnemann/Desktop/Courses/Semester2/Project/Group Project/Code/72sample_noise_data.csv')

# Drop rows with NaN values
cleaned_data = data.dropna(subset=['Noise type', 'BLEU-2'])

# Create boxplots for all the different noise types for BLEU-2
plt.figure(figsize=(12, 8))

# Extract data for BLEU-2 for each noise type
bleu2_scores = [cleaned_data[cleaned_data['Noise type'] == noise_type]['BLEU-2'] for noise_type in sorted(cleaned_data['Noise type'].unique())]

# Plot boxplots
plt.boxplot(bleu2_scores, labels=[f'NT {int(noise_type)}' for noise_type in sorted(cleaned_data['Noise type'].unique())])

plt.title('Boxplot of BLEU-2 Scores by Noise Type')
plt.xlabel('Noise Type')
plt.ylabel('BLEU-2 Score')
plt.grid(True)
plt.show()
