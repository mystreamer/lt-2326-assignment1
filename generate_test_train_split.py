import argparse
import os
import json
from dataclasses import dataclass
import csv
import random
import pprint
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
                    prog='Train-Test-Split Generator',
                    description='This program generates a train-test-split of the ThaiOCR dataset.',
                    epilog='Enjoy the program! :)'
                    )
parser.add_argument('-l',
                    '--languages', 
                    type=str, 
                    default=['en', 'th'], 
                    nargs='+',
                    help='Specify which language(s) should be included in the generated dataset.'
                )
parser.add_argument('--train_set_resolutions',
                    type=str,
                    default='all',
                    nargs='+',
                    help='Specify which resolution(s) should be included in the generated train set, e.g. "200" or "200,300".'
                )
parser.add_argument('--train_set_fontstyles',
                     type=str,
                    default='all',
                    nargs='+',
                    help='Specify which fontstyle(s) should be included in the generated test set, e.g. "italic" or "italic,bold".'
                )
parser.add_argument('--test_set_resolutions',
                    type=str,
                    default='all',
                    nargs='+',
                    help='Specify which resolution(s) should be included in the generated test set, e.g. "200" or "200,300".'
                )
parser.add_argument("--test_set_fontstyles",
                    type=str,
                    default='all',
                    nargs='+',
                    help='Specify which fontstyle(s) should be included in the generated test set, e.g. "italic" or "italic,bold".'
                )
parser.add_argument("--test_set_size",
                    type=float,
                    default=.8,
                    help='How large of a percentage must the test set be?'
                )
parser.add_argument("-iv",
                    "--include_val",
                    default=False,
                    action='store_true',
                    help='Include a validation set in the generated dataset.'
                )
parser.add_argument("--inference",
                    default=False,
                    action='store_true',
                    help='Generate a dataset for inference.')

parser.add_argument("-d",
                    "--debug_mode",
                    default=False,
                    action='store_true'
                )
parser.add_argument("-c",
                    "--config",
                    type=str,
                    default='config.json',
                    help='Specify the path to the config file.'
                )

# Parse the CLI-arguments
args = parser.parse_args()
# print(args)

# Parse the JSON-config file
config_file_contents = {}
if os.path.exists(args.config):
    with open(args.config, 'r') as f:
        config = json.load(f)
        for key, value in config.items():
            config_file_contents[key] = value
# print(config_file_contents)

@dataclass
class Config:
    languages: list
    train_set_resolutions: list
    train_set_fontstyles: list
    test_set_resolutions: list
    test_set_fontstyles: list
    test_set_size: float
    debug_mode: bool
    include_val: bool
    folders: list
    inference: bool = False

config = Config(
    languages=args.languages,
    train_set_fontstyles=args.train_set_fontstyles,
    train_set_resolutions=args.train_set_resolutions,
    test_set_fontstyles=args.test_set_fontstyles,
    test_set_resolutions=args.test_set_resolutions,
    test_set_size=args.test_set_size,
    debug_mode=args.debug_mode,
    include_val=args.include_val,
    folders=config_file_contents['folders'],
    inference=args.inference
)
# print(config)

pp = pprint.PrettyPrinter(indent=4)

# charmap dict
def load_charmap(map_path):
    charmap = {}
    with open(map_path, 'r', encoding='iso-8859-11') as txtfile:
        for i, line in enumerate(txtfile):
            try:
                print(line)
                char_code, char = line.split()
                # if i == 170:
                    # import pdb; pdb.set_trace()
            except ValueError:
                print(f"Could not unpack enough values from line on L{i}")
                continue
            charmap[char_code] = char
    return charmap

charmap = load_charmap(config.folders[0]['charmap'])
# pp.pprint(charmap)
# print(load_charmap(config.folders[1]['charmap']))

def get_char_subfolders(folder_path, folder_range):
    subfolders = []
    for i in range(*folder_range):
        subfolder = os.path.join(folder_path, str(i).zfill(3))
        # make absolute path
        subfolder = os.path.abspath(subfolder)
        if os.path.exists(subfolder):
            subfolders.append(subfolder)
        else:
            print(f"Folder {subfolder} does not exist.")
    return subfolders 
# sf = get_char_subfolders(config.folders[0]["path"], config.folders[0]["char_range"])
# pp.pprint(sf)
# print(get_char_subfolders(config.folders[1]["path"], config.folders[1]["char_range"]))
# print(sf[1])

def get_all_filepaths_in_folder(folder_path):
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.bmp'):
                files.append(os.path.join(root, filename))
            else:
                continue
    return files
# sf1_files = get_all_filepaths_in_folder(sf[1])

# --- Build a dict with all files in the subfolder structured by resolution x font ---
def build_file_dict(files):
    file_dict = {}
    for file in files:
        file_parts = file.split(os.sep)
        resolution = file_parts[-3]
        font = file_parts[-2]
        # initialisers
        if resolution not in file_dict:
            file_dict[resolution] = {}
        if font not in file_dict[resolution]:
            file_dict[resolution][font] = []
        # assign file to the right group
        file_dict[resolution][font].append(file)
    return file_dict
# sf1_files_dict = build_file_dict(sf1_files)
# pp.pprint(sf1_files_dict)

def debbi(fun, *args):
    try:
        res = fun(*args)
    except ValueError:
        import pdb; pdb.set_trace()
    return res

# -- Filter dict by criteria and sample in stratified way printing to a csv file --
def subsample(nested_dict):
    if config.debug_mode:
        # subsample the dict
        return {k: {k2: debbi(random.sample, v2, 5) for k2, v2 in v1.items()} for k, v1 in nested_dict.items()}
filter_resolutions = lambda x : {k: v for k, v in x.items() if k in config.train_set_resolutions}
# sf1_files_filtered_res = filter_resolutions(sf1_files_dict)
# pp.pprint(sf1_files_filtered_res)

filter_fontstyles = lambda x : {k1: {k2: v2 for k2, v2 in v1.items() if k2 in config.train_set_fontstyles} for k1, v1 in x.items()}
# sf1_files_filtered_res_font = filter_fontstyles(sf1_files_filtered_res)
# pp.pprint(sf1_files_filtered_res_font)
# import pdb; pdb.set_trace()

def train_test_val_split(X, test_set_size):
    train, test_val = train_test_split(X, test_size=test_set_size, random_state=42)
    if config.include_val:
        test, val = train_test_split(test_val, test_size=0.5, random_state=42)
        return ("train", train), ("test", test), ("val", val)
    elif config.inference:
        return ("test", X),
    else:
        return ("train", train), ("test", test_val)

    
# we want a stratified train-test-split case
stratified_train_test_split = lambda x : {k1: 
                                          {k2: 
                                           {k3: v3 for k3, v3 in 
                                            train_test_val_split(v2, config.test_set_size)
                                                } 
                                           for k2, v2 in v1.items()} 
                                           for k1, v1 in x.items()
                                           }

def write_to_csv():
    with open(f'./{ "inference/inference_data" if config.inference else "train_test_split" }.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['language', 'char_code', 'char', 'resolution', 'font', 'type', 'fpath']) # write the header
        for language in config.languages:
            config_idx = [idx for idx, folder in enumerate(config.folders) if folder['language'] == language]
            for chars in get_char_subfolders(config.folders[config_idx[0]]['path'], config.folders[config_idx[0]]['char_range']):
                char_code = chars.split(os.sep)[-1]
                char = charmap[char_code]
                res = stratified_train_test_split( 
                        filter_fontstyles(
                            filter_resolutions(
                                subsample(
                                    build_file_dict(
                                        get_all_filepaths_in_folder(chars)
                                    )))))
                for resolution, font_dict in res.items():
                    for font, train_test_dict in font_dict.items():
                        for type, fpath in train_test_dict.items():
                            for f in fpath:
                                print([language, char_code, char, resolution, font, type, f])
                                csv_writer.writerow([language, char_code, char, resolution, font, type, f])
write_to_csv()