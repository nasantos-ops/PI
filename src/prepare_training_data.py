import pandas as pd
import os

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
INPUT_EXCEL  = "data_full.xlsx"           # your Excel file in the project root
OUTPUT_CSV   = "metadata/training_dataset_clean.csv"
OUTPUT_STATS = "metadata/dataset_stats.txt"

# Make sure metadata folder exists
os.makedirs("metadata", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Load the Excel sheet
# ══════════════════════════════════════════════════════════════════════════════
print("Loading Excel sheet...")
df = pd.read_excel(INPUT_EXCEL)
print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"  Columns found: {list(df.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Forward-fill the Folder and Filename columns
# ══════════════════════════════════════════════════════════════════════════════
df['Folder']            = df['Folder'].ffill()
df['Original_Filename'] = df['Original_Filename'].ffill()

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: Clean up Task_Label
# ══════════════════════════════════════════════════════════════════════════════
df['Task_Label'] = df['Task_Label'].str.strip().str.lower()

typo_map = {
    'move_object_bimannual':    'move_object_bimanual',
    'move_object_bimanual ':    'move_object_bimanual',
    'reach_and_grasp ':         'reach_and_grasp',
    'reach and grasp':          'reach_and_grasp',
    'reach_and graso_handle':   'reach_and_grasp_handle',
    'reach_and_grasp_handle ':  'reach_and_grasp_handle',
    'lower_and_place ':         'lower_and_place',
    'lower and place':          'lower_and_place',
    'lower_and_release':        'lower_and_place',
    'pull_towads':              'pull_towards',
    'pull_towards ':            'pull_towards',
    'push_towards ':            'push_towards',
    'grasp_and_twist off':      'grasp_and_twist_off',
    'place_in_bag':             'transport_and_place_in_bag',
    'lift_objects':             'lift_object',
    'lift_object ':             'lift_object',
    'lift_objects ':            'lift_object',
    'dual_slide_and_place ':    'dual_slide_and_place',
    'dual_lift_objects ':       'dual_lift_objects',
    'dual_move_objects ':       'dual_move_objects',
    'engage_latch ':            'engage_latch',
    'fold_and_place ':          'fold_and_place',
    'swing_open ':              'swing_open',
    'close_lid ':               'close_lid',
    'lift_upwards ':            'lift_upwards',
    'pull_downwards ':          'pull_downwards',
    'reach_and_orient ':        'reach_and_orient',
    'stabalize_object':         'stabilize_object',
    'move_object_singular ':    'move_object_singular',
    'move_object_bimanual ':    'move_object_bimanual',
    'retract_hand ':            'retract_hand',
    'retract_hands':            'retract_hand',
    'lower_object ':            'lower_object',
    'slide towards':            'slide_towards',
    'dual_retract_hands':       'retract_hand',
    'reach_for_two_fruits':     'reach_fruit',
    'push_down':                'push_down',
    'Pull_downwards':           'pull_downwards',
    'Push_down':                'push_down',
}
df['Task_Label'] = df['Task_Label'].replace(typo_map)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: Clean binary columns
# ══════════════════════════════════════════════════════════════════════════════
for col in ['Is_Occluded_0_1', 'Is_Bimanual_0_1', 'Is_Fragile_0_1']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: Clean Grip_Type
# ══════════════════════════════════════════════════════════════════════════════
if 'Grip_Type' in df.columns:
    df['Grip_Type'] = df['Grip_Type'].str.strip().str.title().fillna('Unknown')

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6: Build a unique Video_ID per clip
# ══════════════════════════════════════════════════════════════════════════════
df['Video_ID'] = (
    df['Folder'].astype(str).str.strip() + "__" +
    df['Original_Filename'].astype(str).str.strip() + "__" +
    df['Start_Time(s)'].astype(str).str.strip()
)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7: Assign numeric label IDs for ML training
# ══════════════════════════════════════════════════════════════════════════════
unique_tasks = sorted(df['Task_Label'].dropna().unique())
task_to_id   = {task: i for i, task in enumerate(unique_tasks)}
df['Task_Label_ID'] = df['Task_Label'].map(task_to_id)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8: Select final columns
# ══════════════════════════════════════════════════════════════════════════════
wanted_cols = [
    'Video_ID', 'Folder', 'Original_Filename',
    'Start_Time(s)', 'End_Time(s)',
    'Task_Label', 'Task_Label_ID',
    'Is_Occluded_0_1', 'Is_Bimanual_0_1',
    'Grip_Type', 'Material_Type', 'Is_Fragile_0_1', 'Notes',
]
output_cols = [c for c in wanted_cols if c in df.columns]
df_clean = df[output_cols].dropna(subset=['Task_Label'])

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9: Save
# ══════════════════════════════════════════════════════════════════════════════
df_clean.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved clean dataset  → {OUTPUT_CSV}")
print(f"   Rows: {len(df_clean)}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 10: Print stats
# ══════════════════════════════════════════════════════════════════════════════
stats_lines = []
stats_lines.append("=" * 55)
stats_lines.append("  ROBOTICS TRAINING DATASET — SUMMARY")
stats_lines.append("=" * 55)
stats_lines.append(f"  Total segments:      {len(df_clean)}")
stats_lines.append(f"  Unique task labels:  {df_clean['Task_Label'].nunique()}")
if 'Is_Occluded_0_1' in df_clean.columns:
    stats_lines.append(f"  Occluded segments:   {df_clean['Is_Occluded_0_1'].sum()} ({df_clean['Is_Occluded_0_1'].mean()*100:.1f}%)")
if 'Is_Bimanual_0_1' in df_clean.columns:
    stats_lines.append(f"  Bimanual segments:   {df_clean['Is_Bimanual_0_1'].sum()} ({df_clean['Is_Bimanual_0_1'].mean()*100:.1f}%)")
stats_lines.append(f"  Video domains:       {df_clean['Folder'].nunique()}")
stats_lines.append("")
stats_lines.append("  TOP 10 TASK LABELS:")
top_tasks = df_clean['Task_Label'].value_counts().head(10)
for task, count in top_tasks.items():
    stats_lines.append(f"    {task:<38} {count:>4}")
stats_lines.append("")
if 'Grip_Type' in df_clean.columns:
    stats_lines.append("  GRIP TYPE BREAKDOWN:")
    for grip, count in df_clean['Grip_Type'].value_counts().items():
        stats_lines.append(f"    {grip:<20} {count:>4}")
    stats_lines.append("")
stats_lines.append("  TASK LABEL → ID MAPPING:")
for task, tid in task_to_id.items():
    stats_lines.append(f"    {tid:>3}  {task}")
stats_lines.append("=" * 55)

summary = "\n".join(stats_lines)
print("\n" + summary)

with open(OUTPUT_STATS, 'w') as f:
    f.write(summary)
print(f"\n✅ Stats saved          → {OUTPUT_STATS}")