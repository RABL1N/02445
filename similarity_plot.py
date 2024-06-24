import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
data = pd.read_csv('/Users/rasmuslinnemann/Desktop/Courses/Semester2/Project/Group Project/Code/72sample_noise_data.csv')

# Check the data loaded correctly
print(data.head())

# Group the data by Noise type and Prompt index
grouped = data.groupby(['Noise type', 'Prompt index (Categorical)']).mean(numeric_only=True).reset_index()

# Create the plot
plt.figure(figsize=(10, 6))

for noise_type in grouped['Noise type'].unique():
    subset = grouped[grouped['Noise type'] == noise_type]
    plt.plot(subset['Prompt index (Categorical)'], subset['BLEU-2'], marker='o', label=f'Noise Type {int(noise_type)}')

plt.title('Comparison of BLEU-2 by Noise Type')
plt.xlabel('Prompt Index (Categorical)')
plt.ylabel('BLEU Score')
plt.legend(title='Noise Type')
plt.grid(True)
plt.show()


