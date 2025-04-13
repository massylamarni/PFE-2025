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

from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, AddNoise

# Notes:
# * Accelerometer and Gyroscope files recorded at the same time must have the same name.

# Args
parser = argparse.ArgumentParser()
parser.add_argument('-all', action='store_true', help='Full processing')
parser.add_argument('-plot', action='store_true', help='Plot sequences')
args = parser.parse_args()

STARTING_TIMES = [
    datetime.strptime("2025-02-14 16:34:02.752 UTC+01:00", "%Y-%m-%d %H:%M:%S.%f %Z%z"),
    datetime.strptime("2025-02-14 17:00:18.752 UTC+01:00", "%Y-%m-%d %H:%M:%S.%f %Z%z"),
    datetime.strptime("2025-02-14 17:45:46.901 UTC+01:00", "%Y-%m-%d %H:%M:%S.%f %Z%z"),
    datetime.strptime("2025-02-21 09:00:37.457 UTC+01:00", "%Y-%m-%d %H:%M:%S.%f %Z%z"),
    datetime.strptime("2025-02-21 10:46:44.110 UTC+01:00", "%Y-%m-%d %H:%M:%S.%f %Z%z"),
    datetime.strptime("2025-03-07 09:49:32.635 UTC+01:00", "%Y-%m-%d %H:%M:%S.%f %Z%z")
]

stats = {
    "filtered_row_count": 0,
    "empty_class_row_count": 0
}

PARAMS = {
    "align": {
        "enable": None,
        "settings": {
            "mode": "FILTER",
            "modes": ['FILTER', 'NOFILTER'],
            "merge": True
        },
    },
    "normalize": {
        "enable": None,
        "settings": {
            "merge": True
        },
    },
    "classify": {
        "enable": None,
        "settings": {
            "mode": "NONE",
            "modes": ['NONE', 'SEQUENCE'],
            "merge": True
        },
        "get_dump": {
            "enable": None,
            "settings": {
                "mode": "DIRECT",
                "modes": ['DIRECT'],
                "merge": False
            },
        }
    },
    "sequence": {
        "enable": None,
        "settings": {
            "mode": "DIRECT",
            "modes": ['DIRECT', 'PLOT1', 'PLOT2'],
            "merge": False
        },
        "plot": {
            "enable": None,
        },
        "augment": {
            "enable": None,
            "settings": {
                "merge": True
            },
        }
    },
    "print_stats": {
        "enable": 1
    }
}

if (args.all):
    PARAMS = {
    "align": {
        "enable": 1,
        "settings": {
            "mode": "FILTER",
            "modes": ['FILTER', 'NOFILTER'],
            "merge": True
        },
    },
    "normalize": {
        "enable": 1,
        "settings": {
            "merge": True
        },
    },
    "classify": {
        "enable": 1,
        "settings": {
            "mode": "NONE",
            "modes": ['NONE', 'SEQUENCE'],
            "merge": True
        },
        "get_dump": {
            "enable": 1,
            "settings": {
                "mode": "DIRECT",
                "modes": ['DIRECT'],
                "merge": False
            },
        }
    },
    "sequence": {
        "enable": 1,
        "settings": {
            "mode": "DIRECT",
            "modes": ['DIRECT', 'PLOT1', 'PLOT2'],
            "merge": False
        },
        "plot": {
            "enable": 0,
        },
        "augment": {
            "enable": 1,
            "settings": {
                "merge": True
            },
        }
    },
    "print_stats": {
        "enable": 1
    }
}
elif (args.plot):
    PARAMS = {
    "align": {
        "enable": 1,
        "settings": {
            "mode": "FILTER",
            "modes": ['FILTER', 'NOFILTER'],
            "merge": True
        },
    },
    "normalize": {
        "enable": 1,
        "settings": {
            "merge": True
        },
    },
    "classify": {
        "enable": 0,
        "settings": {
            "mode": "NONE",
            "modes": ['NONE', 'SEQUENCE'],
            "merge": True
        },
        "get_dump": {
            "enable": None,
            "settings": {
                "mode": "DIRECT",
                "modes": ['DIRECT'],
                "merge": False
            },
        }
    },
    "sequence": {
        "enable": 1,
        "settings": {
            "mode": "PLOT2",
            "modes": ['DIRECT', 'PLOT1', 'PLOT2'],
            "merge": False
        },
        "plot": {
            "enable": 1,
        },
        "augment": {
            "enable": 1,
            "settings": {
                "merge": True
            },
        }
    },
    "print_stats": {
        "enable": 1
    }
}

def load_dfs_from_folder_path(acc_input_folder_path, gyr_input_folder_path):
    df = {}
    if (gyr_input_folder_path == None):
        for file_name in os.listdir(acc_input_folder_path):
            if file_name.lower().endswith('.csv'):
                df[file_name] = pd.read_csv(os.path.join(acc_input_folder_path, file_name))
    else:
        df = { "acc": {}, "gyr": {} }
        for file_name in os.listdir(acc_input_folder_path):
            if file_name.lower().endswith('.csv'):
                df["acc"][file_name] = pd.read_csv(os.path.join(acc_input_folder_path, file_name))
        for file_name in os.listdir(gyr_input_folder_path):
            if file_name.lower().endswith('.csv'):
                df["gyr"][file_name] = pd.read_csv(os.path.join(gyr_input_folder_path, file_name))
    
    return df

def save_dfs_to_folder_path(df, output_folder_path, file_name):
    os.makedirs(output_folder_path, exist_ok=True)
    acc_output_file_path = os.path.join(output_folder_path, file_name)
    df.to_csv(acc_output_file_path, index=False)
    print(f"Successfuly saved to {acc_output_file_path}")

def write_tree(obj_df, output_folder_path, mode):
    output_folder_path = os.path.join(output_folder_path, mode)
    os.makedirs(output_folder_path, exist_ok=True)
    for key, df in obj_df.items():
        file_name = f"{key}.csv"
        folder_name = 'Unknown'
        if (mode == 'DIRECT'):
            folder_name = ""
        elif (mode == 'PLOT1'):
            split_key = key.split('_', 2)
            folder_name = f"{split_key[0]}_{split_key[1]}"
            if (split_key[2] == '0'):
                os.makedirs(os.path.join(output_folder_path, folder_name), exist_ok=True)
        elif (mode == 'PLOT2'):
            anchor = '-1'
            split_key = key.split('_', 2)
            if (split_key[1] != anchor):
                anchor = split_key[1]
                folder_name = split_key[1]
                os.makedirs(os.path.join(output_folder_path, folder_name), exist_ok=True)
        
        output_file_path = os.path.join(os.path.join(output_folder_path, folder_name), file_name)
        df.to_csv(output_file_path, index=False)
        print(f"Successfuly saved to {output_file_path}")


def align_and_format_dual_df(dual_df, mode, i):
    """Aligns gyroscope data with the accelerometer data and formats the resulting dataframe"""

    def filter_df(df):
        """Filters out Noise"""
        columns_to_exclude = ["DateTime"]
        columns_to_filter = [col for col in df.columns if col not in columns_to_exclude]

        global stats

        for col in columns_to_filter:
            df.loc[:, col] = df[col].replace(0, np.nan)

            stats["filtered_row_count"] += df[col].isna().sum()

            df.loc[:, col] = df[col].interpolate(method='linear')
            df.loc[:, col] = df[col].ffill().bfill()

        return df

    starting_time = STARTING_TIMES[i]
    dual_df['acc']['DateTime'] = dual_df['acc']['Time (s)'].apply(lambda x: starting_time + timedelta(seconds=x))

    # Convert gyroscope timestamps to a NumPy array for faster computation
    gyro_times = dual_df['gyr']['Time (s)'].values
    aligned_rows = []
    for _, accel_row in dual_df['acc'].iterrows():      
        # Align
        time_differences = np.abs(gyro_times - accel_row['Time (s)'])
        closest_gyro_index = np.argmin(time_differences)
        closest_gyro_row = dual_df['gyr'].iloc[closest_gyro_index]

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
    if (mode == 'FILTER'):
        return filter_df(aligned_df)
    else:
        return aligned_df
    
def min_max_normalize_df(df, mode):
    """Normalize values to [0, 1] range"""
    columns_to_exclude = ['DateTime']
    columns_to_normalize = [col for col in df.columns if col not in columns_to_exclude]

    for column in columns_to_normalize:
        min_val = df[column].min()
        max_val = df[column].max()
        df.loc[:, column] = (df[column] - min_val) / (max_val - min_val)
    
    return df

def get_mapped_actions(mode): # Needs rework
    """Maps a JSON file with cattle names and actions"""
    ACTIONS_INPUT_FILE_PATH = 'JSON/actions.json'
    MAPPING_INPUT_FILE_PATH = 'JSON/mapping.json'
    MAP_OUTPUT_FILE_PATH = 'JSON/mapped_actions.json'
    TARGET_SEQUENCE_LENGTH = 14

    with open(ACTIONS_INPUT_FILE_PATH, 'r') as file: actions_file_json = json.load(file)
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
            if (mode == 'NONE'):
                mapped_actions[str(cattle_id)][mapped_action].extend(intervals)
            elif (mode == 'SEQUENCE'):
                for start_str, end_str in intervals:
                    start = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
                    end = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
                    duration = (end - start).total_seconds()

                    center = start + timedelta(seconds=duration / 2)
                    if duration < TARGET_SEQUENCE_LENGTH:
                        half_expand = (TARGET_SEQUENCE_LENGTH - duration) / 2
                        new_start = center - timedelta(seconds=TARGET_SEQUENCE_LENGTH / 2)
                        new_end = center + timedelta(seconds=TARGET_SEQUENCE_LENGTH / 2)
                    else:
                        remainder = duration % TARGET_SEQUENCE_LENGTH
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

def classify_df(df, mode):
    """Add cattle and action column to a dataframe and fill them using a JSON file"""
    df['cattle_id'] = None
    df['action_id'] = None

    mapped_actions = get_mapped_actions(mode)

    for cattle_id, action_ids in mapped_actions.items():
        for action_id, intervals in action_ids.items():
            for interval in intervals:
                df.loc[
                    (df['DateTime'] >= interval[0]) & 
                    (df['DateTime'] <= interval[1]), 
                    ['cattle_id', 'action_id']
                ] = [int(cattle_id), int(action_id)]

    return df

def sequence_df(df, mode):
    """Sequence a dataframe using a JSON file"""
    mapped_actions = get_mapped_actions("NONE")
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

def get_dump(df, mode):
    """Get the unlabeled rows and form a dataset with them"""
    RELEVANT_UNLABELED_SEQUENCE_LENGTH = 600
    empty_class_mask = df['cattle_id'].isna() & df['action_id'].isna()
    
    sequence_indexes = []
    sequence_index = []
    global stats
    
    for i, is_empty_class in enumerate(empty_class_mask):
        if is_empty_class:
            sequence_index.append(i)
        else:
            if len(sequence_index) >= RELEVANT_UNLABELED_SEQUENCE_LENGTH:
                sequence_indexes.append(sequence_index)
            sequence_index = []
    if len(sequence_index) >= RELEVANT_UNLABELED_SEQUENCE_LENGTH: sequence_indexes.append(sequence_index) # Check the last sequence
    
    dump_df = {}
    for seq_num, sequence in enumerate(sequence_indexes):
        seq_df = df.iloc[sequence]
        file_name = f"{seq_num}_{len(sequence)}"
        dump_df[file_name] = seq_df
        stats["empty_class_row_count"] += len(sequence)

    return dump_df

def augment_df(df, mode):
    """Augment small sequences"""
    AUGMENTATION_RATE = 30

    columns_to_exclude = ['DateTime']
    columns_to_process = [col for col in df.columns if col not in columns_to_exclude]

    original_data = df[columns_to_process].values  # Shape: [n_timesteps, 6]
    original_data = original_data.T  # Shape: [6, n_timesteps]

    augmentation = (
        AddNoise(scale=0.01)
        + TimeWarp(n_speed_change=3)  # Random time warping
        # + Crop(size=100)  # Random cropping (adjust size to your sequence length)
        + Quantize(n_levels=20)  # Quantization to simulate sensor precision
        + Drift(max_drift=0.1)  # Add drift to mimic sensor bias over time
        + Reverse(prob=0.3)  # Reverse segments with 30% probability
    )

    augmented_chunks = []

    for _ in range(AUGMENTATION_RATE):
        # Apply different random augmentations each iteration
        augmented = augmentation.augment(original_data)
        augmented_chunks.append(augmented.T)

    augmented_data = np.vstack(augmented_chunks) # Shape: [n_timesteps*AUGMENTATION_RATE, 6]
    augmented_df = pd.DataFrame(augmented_data)

    return augmented_df



def plot_actions():
        """Plot dataframes in nested directories"""
        dfs_input_path = PARAMS_MAP["sequence"]["out_path"]
        OUTPUT_FOLDER_PATH = "PLOT/"

        def plot_action(df, file_name, outpout_folder_path):
            """Plot single dataframe and saves it"""
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

        for root, dirs, files in os.walk(dfs_input_path):
            if root == dfs_input_path:
                continue
            csv_files = [f for f in files if f.lower().endswith('.csv')]

            dfs = []
            for csv_file in csv_files: dfs.append(pd.read_csv(os.path.join(root, csv_file)))
            merged_df = pd.concat(dfs, ignore_index=True)
            
            file_name = os.path.basename(root)
            root = os.path.join(root, OUTPUT_FOLDER_PATH)
            output_path = os.path.join(root, f"{file_name}.csv")
            merged_df.to_csv(output_path, index=False)
            plot_action(merged_df, file_name, root)
            print(f"Plotted {len(csv_files)} files in {root}")

def print_action_stats(df):
    """Prints info about dataframe and some processing stats"""
    print('Stats:')
    dataset_row_count = len(df)
    print(f"Total dataset row count: {dataset_row_count}, ~( {str(timedelta(seconds=dataset_row_count/10)).split('.')[0]} )")
    print(f'Filtered rows count: {stats["filtered_row_count"]}')
    action_id_col_sorted = df['action_id'].value_counts().sort_index()
    
    total_row_count = 0
    for value, count in action_id_col_sorted.items():
        print(f"{value} count: {count}, ~( {str(timedelta(seconds=count/10)).split('.')[0]} )")
        total_row_count += count
    print(f"Total classified row count: {total_row_count}, ~( {str(timedelta(seconds=total_row_count / 10)).split('.')[0]} )")
    print(f"Total unclassified row count: {stats["empty_class_row_count"]}, ~( {str(timedelta(seconds=stats["empty_class_row_count"] / 10)).split('.')[0]} )")

PARAMS_MAP = {
    "align": {
        "ask": "Align ?",
        "progress": "Aligning...",
        "in_path": "",
        "out_path": "Results/Alligned/",
        "task": (align_and_format_dual_df)
    },
    "normalize": {
        "ask": "Normalize ?",
        "progress": "Normalizing...",
        "in_path": "Results/Alligned/",
        "out_path": "Results/Normalized/",
        "task": (min_max_normalize_df)
    },
    "classify": {
        "ask": "Classify ?",
        "progress": "Classifying...",
        "in_path": "Results/Normalized/",
        "out_path": "Results/Classified/",
        "task": (classify_df)
    },
    "get_dump": {
        "ask": "Get dumped dataframe ?",
        "progress": "Getting dumped dataframe...",
        "in_path": "Results/Classified/",
        "out_path": "Results/Dumpdf/",
        "task": (get_dump)
    },
    "sequence": {
        "ask": "Sequence ?",
        "progress": "Sequencing...",
        "in_path": "Results/Normalized/",
        "out_path": "Results/Sequenced/",
        "task": (sequence_df)
    },
    "plot": {
        "ask": "Plot actions ?",
        "progress": "Plotting...",
        "in_path": "Results/Sequenced/",
        "out_path": "Results/Sequenced/",
        "task": (plot_actions)
    },
    "augment": {
        "ask": "Augment small classes ?",
        "progress": "Augmenting...",
        "in_path": "Results/Sequenced/DIRECT/",
        "out_path": "Results/Augmented/",
        "task": (augment_df)
    }
}

def set_params(obj):
    for key, value in obj.items():
        if (key == "settings"):
            if("mode" in value and len(value["modes"]) != 1):
                obj[key]["mode"] = Prompt.ask('mode:', choices=value["modes"], case_sensitive=False, default=value["mode"], show_default=True)
        elif (key != "enable"):
            if("enable" in value and value["enable"] == None):
                if (Prompt.ask(PARAMS_MAP[key]['ask'], choices=['y', 'N'], case_sensitive=False, default='N', show_default=False) == 'y'):
                    obj[key]["enable"] = 1
                    set_params(value)
                else:
                    obj[key]["enable"] = 0
            else:
                set_params(value)

set_params(PARAMS)

def process_params(obj):
    for key, value in obj.items():
        if (key != "enable"):
            if("enable" in value and value["enable"] == 1):
                mode = None
                if ("settings" in value):
                    if ("mode" in value["settings"]): mode = obj[key]["settings"]["mode"]
                    if ("merge" in value["settings"]): merge = obj[key]["settings"]["merge"]

                if (key == 'plot'):
                    plot_actions()
                if (key == 'print_stats'):
                    [(file_name, df)] = load_dfs_from_folder_path(os.path.join(PARAMS_MAP["classify"]["out_path"], "Merged"), None).items()
                    print_action_stats(df)
                elif (key == 'align'):
                    acc_input_folder_path = "Accelerometer/"
                    gyr_input_folder_path = "Gyroscope/"
                    output_folder_path = PARAMS_MAP[key]["out_path"]

                    # Read
                    dual_dfs = load_dfs_from_folder_path(acc_input_folder_path, gyr_input_folder_path)

                    # Process & Write
                    with Progress() as progress:
                        task_progress = progress.add_task(PARAMS_MAP[key]["progress"], total=100)
                        step = 100 / len(dual_dfs['acc'])

                        func = PARAMS_MAP[key]["task"]
                        for i, (file_name, acc_df) in enumerate(dual_dfs["acc"].items()):
                            dual_df = {"acc": acc_df, "gyr": dual_dfs["gyr"][file_name]}
                            result_df = func(dual_df, mode, i)
                            
                            # Write
                            save_dfs_to_folder_path(result_df, output_folder_path, file_name)
                            progress.update(task_progress, advance=step)
                else:
                    input_folder_path = PARAMS_MAP[key]["in_path"]
                    output_folder_path = PARAMS_MAP[key]["out_path"]
                    df_array = []

                    # Read
                    dfs = load_dfs_from_folder_path(input_folder_path, None)

                    # Process and Write
                    with Progress() as progress:
                        task_progress = progress.add_task(PARAMS_MAP[key]["progress"], total=100)
                        step = 100 / len(dfs)

                        func = PARAMS_MAP[key]["task"]
                        for file_name, df in dfs.items():
                            result_df = func(df, mode)
                            df_array.append(result_df)
                            
                            # Write
                            if (key == 'sequence' or key == 'get_dump'):
                                write_tree(result_df, PARAMS_MAP[key]["out_path"], mode)
                            else:                                
                                save_dfs_to_folder_path(result_df, output_folder_path, file_name)
                            progress.update(task_progress, advance=step)

                        if (merge):
                            merged_df = pd.concat(df_array, ignore_index=True)
                            output_folder_path = os.path.join(output_folder_path, "Merged")
                            os.makedirs(output_folder_path, exist_ok=True)
                            save_dfs_to_folder_path(merged_df, output_folder_path, f"{key}_merged.csv")
                process_params(value)

process_params(PARAMS)









