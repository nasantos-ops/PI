import pandas as pd
import os
import subprocess 
from datetime import time

EXCEL_FILE = 'data.xlsx' 


SEARCH_DIRECTORIES = [
    '../GroceryVideos', 
    '../indoorvideos', 
    '../FruitPicking',
    '../Raw_videos'
]
OUTPUT_FOLDER = 'final_clips'
#Makes sure the videos are cut split correctly into folders based on the original video name
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def convert_to_seconds(val):
    """Ensures time format in Excel is correctly converted to total seconds."""
    if isinstance(val, time):
        return float(val.hour * 3600 + val.minute * 60 + val.second)
    try:
        return float(val)
    except:
        return 0.0

print(f"Fresh Start: Processing {EXCEL_FILE}...")

if not os.path.exists(EXCEL_FILE):
    print(f"Error: Could not find {EXCEL_FILE} in this folder!")
else:
    df = pd.read_excel(EXCEL_FILE)
    df['Original_Filename'] = df['Original_Filename'].ffill()
    print(f"Excel file loaded! Found {len(df)} rows.")

    for index, row in df.iterrows():
        video_name = str(row['Original_Filename']).strip()
        
       
        video_folder_name = os.path.splitext(video_name)[0]
        video_output_dir = os.path.join(OUTPUT_FOLDER, video_folder_name)
        
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        start = convert_to_seconds(row['Start_Time(s)'])
        end = convert_to_seconds(row['End_Time(s)'])
        duration = end - start
        label = str(row['Task_Label']).replace(" ", "_")
        
        input_path = None
        for folder in SEARCH_DIRECTORIES:
            path = os.path.join(folder, video_name)
            if os.path.exists(path):
                input_path = path
                break
        
        if input_path:
            output_name = f"row{index}_{label}.mp4"
            output_path = os.path.join(video_output_dir, output_name)
            
            print(f"Row {index}: Cutting {label} for {video_folder_name} ({start}s to {end}s)")
            
            cmd = [
                'ffmpeg', '-y', 
                '-ss', str(start), 
                '-i', input_path, 
                '-t', str(duration), 
                '-c', 'copy', 
                output_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        else:
            print(f"Row {index}: '{video_name}' not found.")

print("\nAll done!")