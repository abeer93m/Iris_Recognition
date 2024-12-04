import pandas as pd
from collections import defaultdict

# Load the CSV file
file_path = '/home/hous/Desktop/IRIS_IDentification/data/filtered_ellipse_data.csv'
df = pd.read_csv(file_path)

# Dictionary to store the total number of images for each user
user_image_count = defaultdict(int)

# Iterate over each row in the DataFrame
for _, row in df.iterrows():
    # Extract the user ID from the image path (e.g., '00075')
    image_path = row['Image Path']
    user_id = image_path.split('/')[-3]  # Assuming the user ID is the 4th element from the end
    
    # Increment the count for the user
    user_image_count[user_id] += 1

# Save the results to a text file
output_file_path = 'user_image_counts.txt'
with open(output_file_path, 'w') as f:
    for user, count in user_image_count.items():
        f.write(f"User {user}: {count} images\n")

print(f"Total number of images for each user has been saved to {output_file_path}")
