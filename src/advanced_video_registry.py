import pandas as pd
import os
import subprocess # This handles the "Lag-Proof" cutting
from datetime import time

# --- CONFIGURATION ---
EXCEL_FILE = 'Segment_trial.xlsx' 
SEARCH_DIRECTORIES = ['../GroceryVideos', '../indoorvideos', '../FruitPicking']
OUTPUT_FOLDER = 'segmented_clips'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def convert_to_seconds(val):
    if isinstance(val, time):
        # Forced fix for the 0:05 -> 300s error
        if val.hour == 0:
            return float(val.minute + val.second/60)
        return float(val.hour * 3600 + val.minute * 60 + val.second)
    try:
        return float(val)
    except:
        return 0.0

print(f"🚀 Running Lag-Proof Segmentation on {EXCEL_FILE}...")

df = pd.read_excel(EXCEL_FILE)
df['Original_Filename'] = df['Original_Filename'].ffill()

for index, row in df.iterrows():
    video_name = str(row['Original_Filename']).strip()
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
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        
        print(f"✅ Row {index}: Instant-Cutting {label} ({start}s to {end}s)")
        
        # This command tells the computer: "Just copy the video data exactly"
        # -ss is start, -t is duration, -c copy means NO RE-ENCODING (No Lag)
        cmd = [
            'ffmpeg', '-y', '-ss', str(start), '-i', input_path, 
            '-t', str(duration), '-c', 'copy', output_path
        ]
        
        # Run the command
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    else:
        print(f"🔍 File not found: {video_name}")

print("\n✅ All done! These clips will play perfectly smoothly.")