import glob
import os
import re

def get_data_files_list(dir_name, file_name_pattern):
    search_pattern = os.path.join(dir_name,
        file_name_pattern.replace("%d", "[0-9]+"))
    all_files_in_files_dir = glob.glob(os.path.join(dir_name, "*"))
    return sorted([f for f in all_files_in_files_dir \
        if re.match(search_pattern, f)])
