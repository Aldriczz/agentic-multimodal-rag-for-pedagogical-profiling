import time

from google_play_scraper import search, app
from others.utils import FileHandler
from others.config import Config

class AppsScraper:
    def get_unique_apps_id_by_keywords(keywords : list) -> set:
        results = set()
        
        for keyword in keywords:
            print(f"Searching for keyword: {keyword}")
            time.sleep(1) 
            for country in Config.COUNTRIES_LIST:
                try:
                    applications = search(
                        keyword,
                        lang='en',
                        country=country,
                        n_hits=200,
                    )
                except Exception as e:
                    continue

            for application in applications:
                results.add(application['appId'])

        print(f"Found unique apps count: {len(results)}")
        return results
    
    def get_similar_apps(app_id : str) -> set:
        results = set()

        for country in Config.COUNTRIES_LIST:
            try:
                target = app(app_id, lang='en', country=country)
                applications = search(
                    target['title'],
                    lang='en',
                    country=country,
                    n_hits=200
                )

                for application in applications:
                    results.add(application['appId'])
            except:
                continue
            
        return results
    
    def get_apps(keywords : list):
        APPS = []
        APP_IDS = set()

        apps_id = AppsScraper.get_unique_apps_id_by_keywords(keywords)

        APP_IDS.update(apps_id)

        for app_id in apps_id:
            print(f"Getting similar apps for {app_id}")
            similar_apps_id = AppsScraper.get_similar_apps(app_id)
            APP_IDS.update(similar_apps_id)
            
        for app_id in APP_IDS:
            app_data = None
            
            for country in Config.COUNTRIES_LIST:
                try:
                    app_data = app(app_id, lang='en', country=country)
                    break 
                except:
                    continue 
                    
            if app_data:
                APPS.append(app_data)
            else:
                print(f"Skipping {app_id}")

        print(f"Got unique apps count: {len(APP_IDS)}")

        FileHandler.save_app_ids_to_txt(APP_IDS)
        FileHandler.save_data_to_json(APPS, filename =  Config.APPS_DATA_FILENAME)
