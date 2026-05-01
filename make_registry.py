import os
import csv

# 1. Folders to include
target_folders = ["GroceryVideos", "indoorvideos"]
output_csv = "master_video_registry.csv"
extensions = ('.mov', '.MOV', '.mp4')

# 2. These headers allow for multiple segments and renaming
headers = ["Folder", "Filename", "New_Filename", "Task_Label", "Start_Time", "End_Time", "Complexity_Score", "Notes"]

print("Creating a fresh registry with Chopping & Renaming columns...")

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    
    for folder in target_folders:
        if os.path.exists(folder):
            for file in sorted(os.listdir(folder)):
                if file.endswith(extensions):
                    # Folder, Filename, New_Name, Task, Start, End, Complexity, Notes
                    writer.writerow([folder, file, "", "", "", "", "", ""])

print(f"Done! Created '{output_csv}'. Now you have columns for Start/End times!")
