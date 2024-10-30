import os
import pandas as pd
import json
from datetime import datetime

twt_dir = 'Datasets/JPM_tweets'
data_list = []

for filename in os.listdir(twt_dir):
    file_path = os.path.join(twt_dir, filename)
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            if isinstance(data, list):
                data_list.extend(data)
            else:
                data_list.append(data)
        except json.JSONDecodeError as e:
            print(f"Warning: JSONDecodeError found in file {filename}. Trying different options...")

            try:
                file.seek(0)
                for line in file:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            data_list.append(entry)
                        except json.JSONDecodeError as e_line:
                            print(f"Skipping line in {filename} due to error: {e_line}.")
            except Exception as e_file:
                print(f"Error reading file {filename}: {e_file}.")

df = pd.DataFrame(data_list)

def convert_date(date_str):
    parsed_date = datetime.strptime(date_str, '%a %b %d %H:%M:%S %z %Y')
    return parsed_date.strftime('%m/%d/%Y')

df['created_at'] = df['created_at'].apply(convert_date)

df.to_csv('Datasets/compiled_tweets.csv', index=False)

