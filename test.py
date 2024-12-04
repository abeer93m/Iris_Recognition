import pandas as pd

# Load the CSV file
file_path = '/home/hous/Desktop/IRIS_IDentification/data/filtered_ellipse_data.csv'
df = pd.read_csv(file_path)

# Change Iris Angle to 0 if it's between 50 and 90
df.loc[(df['Iris Angle'] > 50) & (df['Iris Angle'] < 100), 'Iris Angle'] = 0

# Save the modified DataFrame back to a CSV file
output_file_path = '/home/hous/Desktop/IRIS_IDentification/data/filtered_ellipse_data.csv'
df.to_csv(output_file_path, index=False)

print("Iris angles between 50 and 100 have been set to 0.")