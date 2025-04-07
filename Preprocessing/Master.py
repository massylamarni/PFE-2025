import os
import shutil
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from rich.progress import Progress
from rich.prompt import Prompt

import argparse

import matplotlib.pyplot as plt

# Args
parser = argparse.ArgumentParser()
parser.add_argument('-all', action='store_true', help='Full processing')
parser.add_argument('-plot', action='store_true', help='Plot sequences')
args = parser.parse_args()

# Read
ACCELEROMETER_INPUT_FOLDER_PATH = 'Accelerometer/'
GYROSCOPE_INPUT_FOLDER_PATH = 'Gyroscope/'
JSON_INPUT_FOLDER_PATH = 'JSON/'
ACTION_INPUT_FILE_PATH = 'JSON/actions.json'
MAPPING_INPUT_FILE_PATH = 'JSON/mapping.json'
MERGE_INPUT_FOLDER_PATH = 'Results/ToMerge'

# Write
ALIGN_OUTPUT_FOLDER_PATH = 'Results/Alligned/'
MERGE_OUTPUT_FOLDER_PATH = 'Results/Merged/'
MERGE_OUTPUT_FILE_PATH = 'Results/Merged/Merged.csv'
NORMALIZE_OUTPUT_FOLDER_PATH = 'Results/Normalized/'
MAP_OUTPUT_FILE_PATH = 'JSON/mapped_actions.json'
CLASSIFY_OUTPUT_FOLDER_PATH = 'Results/Classified'
SEQUENCE_FOLDER_PATH = 'Results/Sequenced'

# Modes
INTERVAL_PROCESSING_MODE = None # NONE, SEQUENCE
MULTI_OUTPUT_MODE = None # DIRECT, GROUPEDV1, GROUPEDV2
FILTER_MODE = None # OFF, ON

# Stats
filtered_row_count = 0

STARTING_TIMES = [
    datetime.strptime("2025-02-14 16:34:02.752 UTC+01:00", "%Y-%m-%d %H:%M:%S.%f %Z%z"),
    datetime.strptime("2025-02-14 17:00:18.752 UTC+01:00", "%Y-%m-%d %H:%M:%S.%f %Z%z"),
    datetime.strptime("2025-02-14 17:45:46.901 UTC+01:00", "%Y-%m-%d %H:%M:%S.%f %Z%z"),
    datetime.strptime("2025-02-21 09:00:37.457 UTC+01:00", "%Y-%m-%d %H:%M:%S.%f %Z%z"),
    datetime.strptime("2025-02-21 10:46:44.110 UTC+01:00", "%Y-%m-%d %H:%M:%S.%f %Z%z"),
    datetime.strptime("2025-03-07 09:49:32.635 UTC+01:00", "%Y-%m-%d %H:%M:%S.%f %Z%z")
]

def merge_csv(input_folder_path, output_folder_path):
    """Merge every file in a directory and save it in another"""
    df_array = []
    for file_name in os.listdir(input_folder_path): df_array.append(pd.read_csv(os.path.join(input_folder_path, file_name))) # Read
    merged_df = pd.concat(df_array, ignore_index=True) # Merge
    os.makedirs(output_folder_path, exist_ok=True) # Write
    merged_df.to_csv(MERGE_OUTPUT_FILE_PATH, index=False)
    print(f"Files merged successfuly !")

def get_mapped_actions():
    """Maps a JSON file with cattle names and actions"""
    with open(ACTION_INPUT_FILE_PATH, 'r') as file: actions_file_json = json.load(file)
    with open(MAPPING_INPUT_FILE_PATH, 'r') as file: mapping_file_json = json.load(file)
    mapped_actions = {}

    # Map cattle names
    for cattle_name, action_names in actions_file_json.items():
        cattle_id = mapping_file_json['Cattle'][cattle_name]
        mapped_actions[str(cattle_id)] = {}

        # Map action names
        for action_name, intervals in action_names.items():
            if action_name in mapping_file_json['Exclude']: continue
            else:
                action_id = mapping_file_json['Include'][action_name]
                
            mapped_action = str(action_id)
            if mapped_action not in mapped_actions[str(cattle_id)]:
                mapped_actions[str(cattle_id)][mapped_action] = []
                
            # Add intervals
            if (INTERVAL_PROCESSING_MODE == 'NONE'):
                mapped_actions[str(cattle_id)][mapped_action].extend(intervals)
            elif (INTERVAL_PROCESSING_MODE == 'SEQUENCE'):
                TARGET_LENGTH = 14
                for start_str, end_str in intervals:
                    start = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
                    end = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
                    duration = (end - start).total_seconds()

                    center = start + timedelta(seconds=duration / 2)
                    if duration < TARGET_LENGTH:
                        half_expand = (TARGET_LENGTH - duration) / 2
                        new_start = center - timedelta(seconds=TARGET_LENGTH / 2)
                        new_end = center + timedelta(seconds=TARGET_LENGTH / 2)
                    else:
                        remainder = duration % TARGET_LENGTH
                        new_start = start + timedelta(seconds=remainder / 2)
                        new_end = end - timedelta(seconds=remainder / 2)

                    mapped_actions[str(cattle_id)][mapped_action].append([
                        new_start.strftime('%Y-%m-%d %H:%M:%S'),
                        new_end.strftime('%Y-%m-%d %H:%M:%S')
                    ])

    # Sorting
    for mapped_cattle in mapped_actions:
        for mapped_action in mapped_actions[mapped_cattle]:
            mapped_actions[mapped_cattle][mapped_action].sort(key=lambda x: datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S'))

    with open(MAP_OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(mapped_actions, f, indent=4, ensure_ascii=False)
    print(f"Actions mapped to {MAP_OUTPUT_FILE_PATH}")

    return mapped_actions

def plot_action(df, file_name, outpout_folder_path):
    FIG_HEIGHT = 6
    FIG_WIDTH = 10

    filtered_df = df
    plt.figure(figsize=(FIG_WIDTH+0.01*len(df), FIG_HEIGHT), dpi=150)
    plt.plot(filtered_df.index/10, filtered_df['XA'], marker='o', linestyle='-', color='r')
    plt.plot(filtered_df.index/10, filtered_df['YA'], marker='o', linestyle='-', color='g')
    plt.plot(filtered_df.index/10, filtered_df['ZA'], marker='o', linestyle='-', color='b')
    plt.xlabel('Temp (s)')
    plt.ylabel('Acceleration sur les 3 axes (m/s^2)')
    plt.title('')
    plt.grid(True)
    os.makedirs(os.path.join(outpout_folder_path, 'Accelerometer'), exist_ok=True)
    plt.savefig(os.path.join(os.path.join(outpout_folder_path, 'Accelerometer'), f'{file_name}.png'), bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(FIG_WIDTH+0.01*len(df), FIG_HEIGHT), dpi=150)
    plt.plot(filtered_df.index/10, filtered_df['XG'], marker='o', linestyle='-', color='r')
    plt.plot(filtered_df.index/10, filtered_df['YG'], marker='o', linestyle='-', color='g')
    plt.plot(filtered_df.index/10, filtered_df['ZG'], marker='o', linestyle='-', color='b')
    plt.xlabel('Temp (s)')
    plt.ylabel('Vitesse angulaire sur les 3 axes (rad/s)')
    plt.title('')
    plt.grid(True)
    os.makedirs(os.path.join(outpout_folder_path, 'Gyroscope'), exist_ok=True)
    plt.savefig(os.path.join(os.path.join(outpout_folder_path, 'Gyroscope'), f'{file_name}.png'), bbox_inches='tight')
    # plt.show()

def plot_actions():
    """Plot dataframes in nested directories"""
    for root, dirs, files in os.walk(SEQUENCE_FOLDER_PATH):
        if root == SEQUENCE_FOLDER_PATH:
            continue
        csv_files = [f for f in files if f.lower().endswith('.csv')]

        dfs = []
        for csv_file in csv_files: dfs.append(pd.read_csv(os.path.join(root, csv_file)))
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Save merged file back to the same subfolder
        file_name = os.path.basename(root)
        output_path = os.path.join(root, f"{file_name}.csv")
        merged_df.to_csv(output_path, index=False)
        plot_action(merged_df, file_name, root)
        print(f"Plotted {len(csv_files)} files in {root}")

def filter_df(df):
    """Filters out Noise"""
    filtered_df = df.copy()

    for col in ['XA', 'YA', 'ZA', 'XG', 'YG', 'ZG']:
        # Remove missing data
        filtered_df_col = filtered_df[col].replace(0, np.nan)
        global filtered_row_count
        filtered_row_count += filtered_df_col.isna().sum()
        filtered_df_col.interpolate(method='linear', inplace=True)
        filtered_df_col.ffill().bfill(inplace=True)

        filtered_df[col] = filtered_df_col
    return filtered_df

def align_and_format_df(a_df, g_df, starting_time):
    """Aligns gyroscope data with the accelerometer data and formats the resulting dataframe"""
    a_df['DateTime'] = a_df['Time (s)'].apply(lambda x: starting_time + timedelta(seconds=x))

    # Convert gyroscope timestamps to a NumPy array for faster computation
    gyro_times = g_df['Time (s)'].values
    aligned_rows = []
    for _, accel_row in a_df.iterrows():      
        # Align
        time_differences = np.abs(gyro_times - accel_row['Time (s)'])
        closest_gyro_index = np.argmin(time_differences)
        closest_gyro_row = g_df.iloc[closest_gyro_index]
        # Format
        aligned_row = {
            # 'SecondsElapsed': accel_row['Time (s)'],
            'DateTime': accel_row['DateTime'],
            'XA': accel_row['X (m/s^2)'],
            'YA': accel_row['Y (m/s^2)'],
            'ZA': accel_row['Z (m/s^2)'],
            'XG': closest_gyro_row['X (rad/s)'],
            'YG': closest_gyro_row['Y (rad/s)'],
            'ZG': closest_gyro_row['Z (rad/s)']
        }
        aligned_rows.append(aligned_row)

    aligned_df = pd.DataFrame(aligned_rows)
    if (FILTER_MODE == 'OFF'):
        return aligned_df
    else:
        return filter_df(aligned_df)

def min_max_normalize_df(df, columns_to_exclude):
    """Normalize values to [0, 1] range"""
    normalized_df = df.copy()

    columns_to_normalize = [col for col in df.columns if col not in columns_to_exclude]

    for column in columns_to_normalize:
        min_val = normalized_df[column].min()
        max_val = normalized_df[column].max()
        normalized_df[column] = (normalized_df[column] - min_val) / (max_val - min_val)
    
    return normalized_df

def classify_df(df):
    """Add cattle and action column to a dataframe and fill them using a JSON file"""
    df['cattle_id'] = None
    df['action_id'] = None

    mapped_actions = get_mapped_actions()

    for cattle_id, action_ids in mapped_actions.items():
        for action_id, intervals in action_ids.items():
            for interval in intervals:
                df.loc[
                    (df['DateTime'] >= interval[0]) & 
                    (df['DateTime'] <= interval[1]), 
                    ['cattle_id', 'action_id']
                ] = [int(cattle_id), int(action_id)]

    return df

def sequence_df(df):
    """Sequence a dataframe using a JSON file"""
    mapped_actions = get_mapped_actions()
    sequenced_df = {}

    for cattle_id, action_ids in mapped_actions.items():
        for action_id, intervals in action_ids.items():
            for i, interval in enumerate(intervals):
                filtered_df = df[
                    (df['DateTime'] >= interval[0]) & 
                    (df['DateTime'] <= interval[1])
                ].copy()
                if not filtered_df.empty:
                    key = f"{cattle_id}_{action_id}_{i}"
                    sequenced_df[key] = filtered_df

    return sequenced_df

def print_action_stats():
    df = pd.read_csv(os.path.join(CLASSIFY_OUTPUT_FOLDER_PATH, 'Merged_normalized_classified.csv'))
    print('Stats:')
    print(f'Filtered rows count: {filtered_row_count}')
    action_id_col_sorted = df['action_id'].value_counts().sort_index()
    
    total_row_count = 0
    for value, count in action_id_col_sorted.items():
        print(f"{value} count: {count}, ~( {str(timedelta(seconds=count/10)).split('.')[0]} )")
        total_row_count += count
    print(f"Total row count: {total_row_count}, ~( {str(timedelta(seconds=total_row_count / 10)).split('.')[0]} )")

# Main
def process_csv(a_folder_path, g_folder_path, a_result_folder_path, g_result_folder_path, in_type, out_type, function_name):
    file_dict = {'a': {}, 'g': {}}
    buffer = []
    ACTION_NAME_MAPPING = {
        'align': ['aligned', 'Aligning...'],
        'normalize': ['normalized', 'Normalizing...'],
        'classify': ['classified', 'Classifying...'],
        'sequence': ['sequenced', 'Sequencing...']
    }
    
    # Read input
    for i, file_name in enumerate(os.listdir(a_folder_path)): file_dict['a'][i] = file_name
    if (in_type == 'm'): 
        for i, file_name in enumerate(os.listdir(g_folder_path)): file_dict['g'][i] = file_name
    
    with Progress() as progress:
        task_progress = progress.add_task(ACTION_NAME_MAPPING[function_name][1], total=100)
        step = 100 / len(file_dict['a'])
        for i in file_dict['a']:
            a_df = pd.read_csv(os.path.join(a_folder_path, file_dict['a'][i]))
            if (in_type == 'm'):
                g_df = pd.read_csv(os.path.join(g_folder_path, file_dict['g'][i]))
            
            # Process
            result_df = {}
            if (function_name == 'align'): result_df = align_and_format_df(a_df, g_df, STARTING_TIMES[i])
            elif (function_name == 'normalize'): result_df = min_max_normalize_df(a_df, ['DateTime'])
            elif (function_name == 'classify'): result_df = classify_df(a_df)
            elif (function_name == 'sequence'): result_df = sequence_df(a_df)
            buffer.append(result_df)

            # Write output
            if (in_type == 'm' and out_type == 'm'):
                a_file_name = file_dict['a'][i].split('_', 1)[1].split('.')[0]
                g_file_name = file_dict['g'][i].split('_', 1)[1].split('.')[0]
                os.makedirs(a_result_folder_path, exist_ok=True)
                os.makedirs(g_result_folder_path, exist_ok=True)
                a_output_path = os.path.join(a_result_folder_path, f"{a_file_name}_{ACTION_NAME_MAPPING[function_name][0]}.csv")
                g_output_path = os.path.join(g_result_folder_path, f"{g_file_name}_{ACTION_NAME_MAPPING[function_name][0]}.csv")
                result_df['a'].to_csv(a_output_path, index=False)
                result_df['g'].to_csv(g_output_path, index=False)
                print(f"{file_dict['a'][i]} {ACTION_NAME_MAPPING[function_name][0]} successfuly !")
                print(f"{file_dict['g'][i]} {ACTION_NAME_MAPPING[function_name][0]} successfuly !")
            elif (in_type == 'm' and out_type == 's'):
                file_name = file_dict['a'][i].split('_', 1)[1].split('.')[0]
                os.makedirs(a_result_folder_path, exist_ok=True)
                output_path = os.path.join(a_result_folder_path, f"{file_name}_{ACTION_NAME_MAPPING[function_name][0]}.csv")
                result_df.to_csv(output_path, index=False)
                print(f"{file_dict['a'][i]} and {file_dict['g'][i]} {ACTION_NAME_MAPPING[function_name][0]} successfuly !")
            elif (in_type == 's' and out_type == 's'):
                file_name = file_dict['a'][i].split('.')[0]
                os.makedirs(a_result_folder_path, exist_ok=True)
                output_path = os.path.join(a_result_folder_path, f"{file_name}_{ACTION_NAME_MAPPING[function_name][0]}.csv")
                result_df.to_csv(output_path, index=False)
                print(f"{file_dict['a'][i]} {ACTION_NAME_MAPPING[function_name][0]} successfuly !")
            else:
                os.makedirs(a_result_folder_path, exist_ok=True)

                for key, df in result_df.items():
                    file_name = f"{key}.csv"
                    if (MULTI_OUTPUT_MODE == 'DIRECT'):
                        output_path = os.path.join(a_result_folder_path, file_name)
                    elif (MULTI_OUTPUT_MODE == 'GROUPEDV1'):
                        split_key = key.split('_', 2)
                        folder_name = f"{split_key[0]}_{split_key[1]}"
                        if (split_key[2] == '0'):
                            os.makedirs(os.path.join(a_result_folder_path, folder_name), exist_ok=True)
                        output_path = os.path.join(os.path.join(a_result_folder_path, folder_name), file_name)
                    elif (MULTI_OUTPUT_MODE == 'GROUPEDV2'):
                        anchor = '-1'
                        split_key = key.split('_', 2)
                        folder_name = 'Unknown'
                        if (split_key[1] != anchor):
                            anchor = split_key[1]
                            folder_name = split_key[1]
                            os.makedirs(os.path.join(a_result_folder_path, folder_name), exist_ok=True)
                        output_path = os.path.join(os.path.join(a_result_folder_path, folder_name), file_name)
                    df.to_csv(output_path, index=False)
                    print(f"{file_name} {ACTION_NAME_MAPPING[function_name][0]} successfuly !")

            progress.update(task_progress, advance=step)
        
        return buffer

aligned_dfs = {}
merged_df = {}
normalized_df = {}
classified_df = {}
sequenced_df = {}
if os.path.exists(MERGE_INPUT_FOLDER_PATH):
    shutil.rmtree(MERGE_INPUT_FOLDER_PATH)
os.makedirs(MERGE_INPUT_FOLDER_PATH)

PARAMS = {}
if (args.all):
    PARAMS = {
        'align': True,
        'merge': True,
        'normalize': True,
        'classify': True,
        'sequence': True,
        'plot_actions': False,
        'show_stats': True
    }
    INTERVAL_PROCESSING_MODE = 'SEQUENCE'
    MULTI_OUTPUT_MODE = 'DIRECT'
    FILTER_MODE = 'ON'
if (args.plot):
    PARAMS = {
        'align': True,
        'merge': True,
        'normalize': True,
        'classify': False,
        'sequence': True,
        'plot_actions': True,
        'show_stats': False
    }
    INTERVAL_PROCESSING_MODE = 'SEQUENCE'
    MULTI_OUTPUT_MODE = 'GROUPEDV2'
    FILTER_MODE = 'ON'

if (FILTER_MODE == None):
    FILTER_MODE = Prompt.ask('Choose filter mode:', choices=['ON', 'OFF'], case_sensitive=False, default='ON', show_default=True)
if (not 'align' in PARAMS):
    if (Prompt.ask('Align ?', choices=['y', 'N'], case_sensitive=False, default='N', show_default=False) == 'y'):
        PARAMS['align'] = True
    else:
        PARAMS['align'] = False
if (not 'merge' in PARAMS):
    if (Prompt.ask('Merge ?', choices=['y', 'N'], case_sensitive=False, default='N', show_default=False) == 'y'):
        PARAMS['merge'] = True
    else:
        PARAMS['merge'] = False
if (not 'normalize' in PARAMS):
    if (Prompt.ask('Normalize ?', choices=['y', 'N'], case_sensitive=False, default='N', show_default=False) == 'y'):
        PARAMS['normalize'] = True
    else:
        PARAMS['normalize'] = False
if (not 'classify' in PARAMS):
    if (Prompt.ask('Classify ?', choices=['y', 'N'], case_sensitive=False, default='N', show_default=False) == 'y'):
        PARAMS['classify'] = True
    else:
        PARAMS['classify'] = False
if (not 'sequence' in PARAMS):
    if (Prompt.ask('Sequence ?', choices=['y', 'N'], case_sensitive=False, default='N', show_default=False) == 'y'):
        PARAMS['sequence'] = True
    else:
        PARAMS['sequence'] = False
if (INTERVAL_PROCESSING_MODE == None and (PARAMS['classify'] or PARAMS['sequence'])):
    INTERVAL_PROCESSING_MODE = Prompt.ask('Choose interval processing mode:', choices=['NONE', 'SEQUENCE'], case_sensitive=False, default='SEQUENCE', show_default=True)
if (MULTI_OUTPUT_MODE == None and (PARAMS['classify'] or PARAMS['sequence'])):
    MULTI_OUTPUT_MODE = Prompt.ask('Choose multi output mode:', choices=['DIRECT', 'GROUPEDV1', 'GROUPEDV2'], case_sensitive=False, default='DIRECT', show_default=True)

if (not 'plot_actions' in PARAMS):
    if (Prompt.ask('Plot actions ?', choices=['y', 'N'], case_sensitive=False, default='N', show_default=False) == 'y'):
        PARAMS['plot_actions'] = True
    else:
        PARAMS['plot_actions'] = False

if (not 'show_stats' in PARAMS):
    if (Prompt.ask('Show stats ?', choices=['y', 'N'], case_sensitive=False, default='N', show_default=False) == 'y'):
        PARAMS['show_stats'] = True
    else:
        PARAMS['show_stats'] = False


if (PARAMS['align']):
    aligned_dfs = process_csv(ACCELEROMETER_INPUT_FOLDER_PATH, GYROSCOPE_INPUT_FOLDER_PATH, ALIGN_OUTPUT_FOLDER_PATH, None, 'm', 's', 'align')
    for file_name in os.listdir(ALIGN_OUTPUT_FOLDER_PATH): shutil.copy2(os.path.join(ALIGN_OUTPUT_FOLDER_PATH, file_name), MERGE_INPUT_FOLDER_PATH)

if (PARAMS['merge']):
    merged_df = merge_csv(MERGE_INPUT_FOLDER_PATH, MERGE_OUTPUT_FOLDER_PATH)

if (PARAMS['normalize']):
    normalized_df = process_csv(MERGE_OUTPUT_FOLDER_PATH, None, NORMALIZE_OUTPUT_FOLDER_PATH, None, 's', 's', 'normalize')

if (PARAMS['classify']):
    classified_df = process_csv(NORMALIZE_OUTPUT_FOLDER_PATH, None, CLASSIFY_OUTPUT_FOLDER_PATH, None, 's', 's', 'classify')

if (PARAMS['sequence']):
    sequenced_df = process_csv(NORMALIZE_OUTPUT_FOLDER_PATH, None, SEQUENCE_FOLDER_PATH, None, 's', 'm', 'sequence')

if (PARAMS['plot_actions']):
    plot_actions()

if (PARAMS['show_stats']):
    print_action_stats()

print('End !')
