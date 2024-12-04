import csv
import cv2

def is_iris_axis_too_large(image_path, iris_major_axis, iris_minor_axis):
    """Check if the iris major or minor axis is greater than half the image's dimensions."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not load image: {image_path}")
        return False
    
    height, width = image.shape[:2]
    
    return iris_major_axis > width / 2 or iris_minor_axis > height / 2

def process_csv_file(input_csv, output_csv, log_file):
    """Process the CSV file to filter out rows with excessively large iris axes."""
    with open(input_csv, 'rb') as f_in:
        data = f_in.read().replace(b'\x00', b'')
    
    with open(output_csv, 'w', newline='', encoding='ISO-8859-1') as outfile, open(log_file, 'w', newline='', encoding='ISO-8859-1') as logfile:
        reader = csv.DictReader(data.decode('ISO-8859-1').splitlines())
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        log_writer = csv.writer(logfile)

        writer.writeheader()
        log_writer.writerow(["Image Path", "Pupil Center X", "Pupil Center Y", "Pupil Axis Major", "Pupil Axis Minor", "Pupil Angle", 
                             "Iris Center X", "Iris Center Y", "Iris Axis Major", "Iris Axis Minor", "Iris Angle"])

        for row in reader:
            image_path = row['Image Path']
            iris_major_axis = float(row['Iris Axis Major'])
            iris_minor_axis = float(row['Iris Axis Minor'])
            
            if is_iris_axis_too_large(image_path, iris_major_axis, iris_minor_axis):
                # Log the problematic row
                log_writer.writerow(row.values())
            else:
                # Write the valid row to the output CSV
                writer.writerow(row)

    print(f"Processing complete. Output saved to {output_csv}. Issues logged in {log_file}.")

def main():
    input_csv = "D:/Iris Identification/IRIS_IDentification/data/ellipse_data_100.csv"  # Path to the input CSV file
    output_csv = "D:/Iris Identification/IRIS_IDentification/data/filtered_ellipse_data.csv"  # Path to the output CSV file
    log_file = "D:/Iris Identification/IRIS_IDentification/data/log_large_iris_axes.csv"  # Path to log file for rows with large axes

    process_csv_file(input_csv, output_csv, log_file)

if __name__ == "__main__":
    main()