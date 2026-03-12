import json
import csv

from datetime import datetime
from others.config import Config

class FileHandler:
    def datetime_converter(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError("Type not serializable")

    # json
    def save_data_to_json(data, filename = Config.APPS_DATA_FILENAME):
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4, default=FileHandler.datetime_converter)
        print(f"Data saved to {filename}")

    def load_data_from_json(filename = Config.APPS_DATA_FILENAME):
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
        print(f"Data loaded from {filename}")
        
    # csv
    def load_data_from_csv(filename = Config.APPS_IMAGE_CAPTIONS_FILENAME):
        with open(filename, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            return [row for row in reader]
        print(f"Data loaded from {filename}")

    # txt
    def save_app_ids_to_txt(app_ids, filename = Config.APPS_ID_FILENAME):
        with open(filename, "a") as file:
            for app_id in app_ids:
                file.write(str(app_id) + "\n")
        print(f"App IDs saved to {filename}")

    def load_app_ids_from_txt(filename = Config.APPS_ID_FILENAME):
        with open(filename, "r") as file:
            app_ids = [line.strip() for line in file]
        return app_ids
        print(f"App IDs loaded from {filename}")