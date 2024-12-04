import os
import shutil

def validate_user_data(image_dir):
    subjects = os.listdir(image_dir)
    for subject in subjects:
        subject_path = os.path.join(image_dir, subject)
        
        # Check if it's a directory
        if os.path.isdir(subject_path):
            valid = True
            
            # Check if both L and R subfolders exist
            for eye in ['L', 'R']:
                eye_path = os.path.join(subject_path, eye)
                if not os.path.exists(eye_path) or not os.listdir(eye_path):
                    valid = False
            
            # If invalid, remove the entire subject directory
            if not valid:
                print(f"Removing invalid directory: {subject_path}")
                shutil.rmtree(subject_path)

def main():
    input_dir = "data/raw"
    validate_user_data(input_dir)
    print("Validation completed. Invalid directories have been removed.")

if __name__ == "__main__":
    main()
