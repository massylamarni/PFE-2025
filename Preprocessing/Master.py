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

from collections import defaultdict


# Notes:
# * Accelerometer and Gyroscope files recorded at the same time must have the same name.
#
# TODO:
# * The interactive parameter selection method allows unstable combinations, must handle either by
# hiding conflicting parameters or by satisfying the dependencies
# * Show comparison graph between original data vs augmented data

# Args
parser = argparse.ArgumentParser()
parser.add_argument('-all', action='store_true', help='Full processing')
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
    "df_row_count": 0,
    "filtered_row_count": 0,
    "empty_class_row_count": 0,
    "original_class_row_count": 0,
    "augmented_class_row_count": 0
}
cached_mapped_actions = {}
segments_info = {}

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
                "merge": False
            },
        }
    },
    "segment": {
        "enable": None,
        "settings": {
            "merge": False
        },
        "plot_original": {
            "enable": None,
        },
        "augment": {
            "enable": None,
            "settings": {
                "merge": True
            },
            "plot_augmented": {
                "enable": None,
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
                "modes": ['FILTER'],
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
                "modes": ['NONE'],
                "merge": True
            },
            "get_dump": {
                "enable": 1,
                "settings": {
                    "merge": False
                },
            }
        },
        "segment": {
            "enable": 1,
            "settings": {
                "merge": False
            },
            "plot_original": {
                "enable": 1,
            },
            "augment": {
                "enable": 1,
                "settings": {
                    "merge": True
                },
                "plot_original": {
                    "enable": 1,
                },
            }
        },
        "print_stats": {
            "enable": 1
        }
    }

def load_dfs_from_folder_path(input_folder_path_1, input_folder_path_2):
    df = {}
    if (input_folder_path_2 == None):
        for file_name in os.listdir(input_folder_path_1):
            if file_name.lower().endswith('.csv'):
                df[file_name] = pd.read_csv(os.path.join(input_folder_path_1, file_name))
    else:
        df = { "df1": {}, "df2": {} }
        for file_name in os.listdir(input_folder_path_1):
            if file_name.lower().endswith('.csv'):
                df["df1"][file_name] = pd.read_csv(os.path.join(input_folder_path_1, file_name))
        for file_name in os.listdir(input_folder_path_2):
            if file_name.lower().endswith('.csv'):
                df["df2"][file_name] = pd.read_csv(os.path.join(input_folder_path_2, file_name))
    
    return df

def save_dfs_to_folder_path(df, output_folder_path, file_name):
    os.makedirs(output_folder_path, exist_ok=True)
    acc_output_file_path = os.path.join(output_folder_path, file_name)
    df.to_csv(acc_output_file_path, index=False)
    print(f"Successfuly saved to {acc_output_file_path}")

def write_tree(obj_df, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)
    for key, df in obj_df.items():
        file_name = f"{key}.csv"
        output_file_path = os.path.join(output_folder_path, file_name)
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
    dual_df['df1']['DateTime'] = dual_df['df1']['Time (s)'].apply(lambda x: starting_time + timedelta(seconds=x))

    # Convert gyroscope timestamps to a NumPy array for faster computation
    gyro_times = dual_df['df2']['Time (s)'].values
    aligned_rows = []
    for _, accel_row in dual_df['df1'].iterrows():      
        # Align
        time_differences = np.abs(gyro_times - accel_row['Time (s)'])
        closest_gyro_index = np.argmin(time_differences)
        closest_gyro_row = dual_df['df2'].iloc[closest_gyro_index]

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

    # with open(MAP_OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f: json.dump(mapped_actions, f, indent=4, ensure_ascii=False)
    print(f"Actions mapped to {MAP_OUTPUT_FILE_PATH}")

    return mapped_actions

def classify_df(df, mode):
    """Add cattle and action column to a dataframe and fill them using a JSON file"""
    df['cattle_id'] = None
    df['action_id'] = None

    global cached_mapped_actions
    if (not mode in cached_mapped_actions): cached_mapped_actions[mode] = get_mapped_actions(mode)
    mapped_actions = cached_mapped_actions[mode]

    for cattle_id, action_ids in mapped_actions.items():
        for action_id, intervals in action_ids.items():
            for interval in intervals:
                mask = (df['DateTime'] >= interval[0]) & (df['DateTime'] <= interval[1])
                df.loc[mask, ['cattle_id', 'action_id']] = [int(cattle_id), int(action_id)]

    return df

def segment_df(df, mode):
    """Segment a dataframe using a JSON file"""

    global cached_mapped_actions
    global segments_info
    if (not "NONE" in cached_mapped_actions): cached_mapped_actions["NONE"] = get_mapped_actions("NONE")
    mapped_actions = cached_mapped_actions["NONE"]
    segmented_df = {}

    df['cattle_id'] = None
    df['action_id'] = None

    for cattle_id, action_ids in mapped_actions.items():
        for action_id, intervals in action_ids.items():
            for i, interval in enumerate(intervals):
                mask = (df['DateTime'] >= interval[0]) & (df['DateTime'] <= interval[1])
                df.loc[mask, ['cattle_id', 'action_id']] = [int(cattle_id), int(action_id)]

                filtered_df = df[mask].copy()
                if not filtered_df.empty:
                    key = f"{action_id}_{cattle_id}_{i}"
                    segmented_df[key] = filtered_df

                    if (not action_id in segments_info): segments_info[action_id] = 0
                    segments_info[action_id] += 1

    return segmented_df

def get_dump(df, mode):
    """Get the unlabeled rows and form a dataset with them"""
    RELEVANT_UNLABELED_SEQUENCE_LENGTH = 600
    empty_class_mask = df['cattle_id'].isna() & df['action_id'].isna()
    
    segment_indexes = []
    segment_index = []
    global stats
    
    for i, is_empty_class in enumerate(empty_class_mask):
        if is_empty_class:
            segment_index.append(i)
        else:
            if len(segment_index) >= RELEVANT_UNLABELED_SEQUENCE_LENGTH:
                segment_indexes.append(segment_index)
            segment_index = []
    if len(segment_index) >= RELEVANT_UNLABELED_SEQUENCE_LENGTH: segment_indexes.append(segment_index) # Check the last segment
    
    dump_df = {}
    for seq_num, segment in enumerate(segment_indexes):
        seq_df = df.iloc[segment]
        file_name = f"{seq_num}_{len(segment)}"
        dump_df[file_name] = seq_df
        stats["empty_class_row_count"] += len(segment)

    return dump_df

def augment_df(df, factor):
    """Augment small segments"""
    AUGMENTATION_RATE = min(factor, 30)
    if (AUGMENTATION_RATE == 0): return None

    columns_to_exclude = ['DateTime']
    columns_to_process = [col for col in df.columns if col not in columns_to_exclude]

    original_data = df[columns_to_process].values  # Shape: [n_timesteps, 6]
    original_data = original_data.T  # Shape: [6, n_timesteps]

    augmentation = (
        AddNoise(scale=0.01)
        + TimeWarp(n_speed_change=3)  # Random time warping
        # + Crop(size=100)  # Random cropping (adjust size to your segment length)
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
    augmented_df = pd.DataFrame(augmented_data, columns=["XA", "YA", "ZA", "XG", "YG", "ZG", "cattle_id", "action_id"])

    return augmented_df

def plot_from_path(input_folder_path, output_folder_path, progress, task_progress):
    """Plot both individual files and merged versions by action_id"""
    
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
            plt.savefig(os.path.join(os.path.join(outpout_folder_path, 'Accelerometer'), f'{file_name}.png'), bbox_inches='tight')
            # plt.show()
            plt.close()

            plt.figure(figsize=(FIG_WIDTH+0.01*len(df), FIG_HEIGHT), dpi=150)
            plt.plot(filtered_df.index/10, filtered_df['XG'], marker='o', linestyle='-', color='r')
            plt.plot(filtered_df.index/10, filtered_df['YG'], marker='o', linestyle='-', color='g')
            plt.plot(filtered_df.index/10, filtered_df['ZG'], marker='o', linestyle='-', color='b')
            plt.xlabel('Temp (s)')
            plt.ylabel('Vitesse angulaire sur les 3 axes (rad/s)')
            plt.title('')
            plt.grid(True)
            plt.savefig(os.path.join(os.path.join(outpout_folder_path, 'Gyroscope'), f'{file_name}.png'), bbox_inches='tight')
            # plt.show()
            plt.close()
    
    file_groups = defaultdict(list)
    
    for file_name in os.listdir(input_folder_path):
        if file_name.endswith('.csv'):
            if file_name.startswith('augment_'):
                action_id = file_name.split('_')[1]
            else:
                action_id = file_name.split('_')[0]
            file_groups[action_id].append((file_name, os.path.join(input_folder_path, file_name)))
    
    os.makedirs(output_folder_path, exist_ok=True)
    step = 100 / len(file_groups)
    for action_id, file_group in file_groups.items():
        action_folder = os.path.join(output_folder_path, action_id)
        os.makedirs(action_folder, exist_ok=True)
        os.makedirs(os.path.join(action_folder, "Accelerometer"), exist_ok=True)
        os.makedirs(os.path.join(action_folder, "Gyroscope"), exist_ok=True)
        
        all_dfs = []
        for file_name, file_path in file_group:
            df = pd.read_csv(file_path)
            all_dfs.append(df)
            base_name = os.path.splitext(file_name)[0]

            plot_action(df, base_name, action_folder)
        
        if len(all_dfs) > 1:
            merged_df = pd.concat(all_dfs, ignore_index=True)
            
            plot_action(merged_df, action_id, action_folder)
        
        print(f"Plotted action_id {action_id} ({len(file_group)} files)")
        progress.update(task_progress, advance=step)

def set_action_stats(df, type):
    """Set action stats"""
    global stats

    df_row_count = len(df)    
    action_id_col_sorted = df['action_id'].value_counts().sort_index()
    if (type == 'original'):
        stats["df_row_count"] = df_row_count
        stats["original_class_row_count"] = action_id_col_sorted
    else:
        stats["augmented_class_row_count"] = action_id_col_sorted

def print_stats():
    """Prints info about dataframe and some processing stats"""
    print('---Processed:')
    print(f'Filtered rows count: {stats["filtered_row_count"]}')
    print(f"Total unclassified row count: {stats["empty_class_row_count"]}, ~( {str(timedelta(seconds=stats["empty_class_row_count"] / 10)).split('.')[0]} )")
    print('---Stats:')
    total_row_count = 0
    for action_id, count in stats["original_class_row_count"].items():
        print(f"Class {int(action_id)} original count: {count}, ~( {str(timedelta(seconds=count/10)).split('.')[0]} )")
        total_row_count += count
    if not (isinstance(stats["augmented_class_row_count"], int) and stats["augmented_class_row_count"] == 0):
        for action_id, count in stats["augmented_class_row_count"].items():
            print(f"Class {int(action_id)} augmented count: {count}, ~( {str(timedelta(seconds=count/10)).split('.')[0]} )")

    print(f"Original dataset row count: {stats['df_row_count']}, ~( {str(timedelta(seconds=stats['df_row_count']/10)).split('.')[0]} )")
    print(f"Total classified row count: {total_row_count}, ~( {str(timedelta(seconds=total_row_count / 10)).split('.')[0]} )")


PARAMS_MAP = {
    "align": {
        "ask": "Align ?",
        "progress": "Aligning...",
        "in_path_1": "Accelerometer/",
        "in_path_2": "Gyroscope/",
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
    "segment": {
        "ask": "Segment ?",
        "progress": "Segmenting...",
        "in_path": "Results/Normalized/",
        "out_path": "Results/Segmented/",
        "task": (segment_df)
    },
    "plot_original": {
        "ask": "Plot actions on the original dataframe ?",
        "progress": "Plotting...",
        "in_path": "Results/Segmented/",
        "out_path": "Results/Segmented/Plot",
        "task": (plot_from_path)
    },
    "augment": {
        "ask": "Augment small classes ?",
        "progress": "Augmenting...",
        "in_path": "Results/Segmented/",
        "out_path": "Results/Augmented",
        "task": (augment_df)
    },
    "plot_augmented": {
        "ask": "Plot actions on the augmented dataframe ?",
        "progress": "Plotting...",
        "in_path": "Results/Final/",
        "out_path": "Results/Final/Plot",
        "task": (plot_from_path)
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

                if (key == 'plot_original'):
                    with Progress() as progress:
                        task_progress = progress.add_task(PARAMS_MAP[key]["progress"], total=100)
                        plot_from_path(PARAMS_MAP["plot_original"]["in_path"], PARAMS_MAP["plot_original"]["out_path"], progress, task_progress)
                elif (key == 'plot_augmented'):
                    
                    with Progress() as progress:
                        task_progress = progress.add_task(PARAMS_MAP[key]["progress"], total=100)
                        plot_from_path(PARAMS_MAP["plot_augmented"]["in_path"], PARAMS_MAP["plot_augmented"]["out_path"], progress, task_progress)
                elif (key == 'print_stats'):
                    [(orignial_file_name, original_df)] = load_dfs_from_folder_path(os.path.join(PARAMS_MAP["classify"]["out_path"], "Merged"), None).items()
                    set_action_stats(original_df, "original")
                    
                    if (PARAMS["segment"]["augment"]["enable"] == 1):
                        [(augmented_file_name, augmented_df)] = load_dfs_from_folder_path(os.path.join(PARAMS_MAP["augment"]["out_path"], "Merged"), None).items()
                        set_action_stats(augmented_df, "augmented")

                    print_stats()

                elif (key == 'align'):
                    input_folder_path_1 = PARAMS_MAP[key]["in_path_1"]
                    input_folder_path_2 = PARAMS_MAP[key]["in_path_2"]
                    output_folder_path = PARAMS_MAP[key]["out_path"]

                    # Read
                    dual_dfs = load_dfs_from_folder_path(input_folder_path_1, input_folder_path_2)

                    # Process & Write
                    with Progress() as progress:
                        task_progress = progress.add_task(PARAMS_MAP[key]["progress"], total=100)
                        step = 100 / len(dual_dfs['df1'])

                        func = PARAMS_MAP[key]["task"]
                        for i, (df1_file_name, df1) in enumerate(dual_dfs["df1"].items()):
                            if (df1_file_name in dual_dfs["df2"]):
                                dual_df = {"df1": df1, "df2": dual_dfs["df2"][df1_file_name]}
                            else:
                                dual_df = {"df1": df1, "df2": None}

                            result_df = func(dual_df, mode, i)
                            
                            # Write
                            save_dfs_to_folder_path(result_df, output_folder_path, df1_file_name)
                            progress.update(task_progress, advance=step)
                else:
                    input_folder_path = PARAMS_MAP[key]["in_path"]
                    output_folder_path = PARAMS_MAP[key]["out_path"]
                    df_array = []

                    if (key == "augment"):
                        [(file_name, df)] = load_dfs_from_folder_path(os.path.join(PARAMS_MAP["classify"]["out_path"], "Merged"), None).items()
                        action_id_col_sorted = df['action_id'].value_counts().sort_index()
                        target_count = np.max(action_id_col_sorted)

                        factors = {}
                        for action_id, count in action_id_col_sorted.items():
                            factors[int(action_id)] = round(target_count / count) - 1
                        
                        specific_factors = {}
                        i = 0
                        sorted_segments_info = {
                            k: v for k, v in sorted(segments_info.items(), key=lambda item: item[0][0])
                        }
                        for action_id, segments_count in sorted_segments_info.items():
                            for j in range(segments_count):
                                specific_factors[i] = factors[int(action_id)]
                                i += 1

                    # Read
                    dfs = load_dfs_from_folder_path(input_folder_path, None)

                    # Process and Write
                    with Progress() as progress:
                        task_progress = progress.add_task(PARAMS_MAP[key]["progress"], total=100)
                        step = 100 / len(dfs)

                        func = PARAMS_MAP[key]["task"]
                        for i, (file_name, df) in enumerate(dfs.items()):
                            if (key == "augment"):
                                result_df = func(df, specific_factors[i])
                                file_name = f"{key}_{file_name}"
                            else:
                                result_df = func(df, mode)
                            if (result_df is None):
                                progress.update(task_progress, advance=step)
                                continue
                            df_array.append(result_df)
                            
                            # Write
                            if (key == 'segment' or key == 'get_dump'):
                                write_tree(result_df, PARAMS_MAP[key]["out_path"])
                            else:
                                save_dfs_to_folder_path(result_df, output_folder_path, file_name)
                            progress.update(task_progress, advance=step)

                    # Postprocessing
                    if (key == 'augment'):
                        original_df_path = PARAMS_MAP["segment"]["out_path"]
                        augmented_df_path = PARAMS_MAP["augment"]["out_path"]
                        final_output_folder_path = PARAMS_MAP["plot_augmented"]["in_path"]
                        os.makedirs(final_output_folder_path, exist_ok=True)
                        for file_name in os.listdir(original_df_path):
                            if file_name.lower().endswith('.csv'):
                                shutil.copy2(os.path.join(original_df_path, file_name), final_output_folder_path)
                        for file_name in os.listdir(augmented_df_path):
                            if file_name.lower().endswith('.csv'):
                                shutil.copy2(os.path.join(augmented_df_path, file_name), final_output_folder_path)

                    if (merge):
                        merged_df = pd.concat(df_array, ignore_index=True)
                        output_folder_path = os.path.join(output_folder_path, "Merged")
                        os.makedirs(output_folder_path, exist_ok=True)
                        save_dfs_to_folder_path(merged_df, output_folder_path, f"{key}_merged.csv")
                process_params(value)

process_params(PARAMS)









