from google_play_scraper import reviews, Sort
from others.utils import FileHandler
from others.config import Config
from tqdm import tqdm

class ReviewsScraper:
    def get_reviews_by_app_id(apps : list) -> list:
        all_apps_reviews = []

        for app in tqdm(apps):
            app_id = app['appId']

            app_reviews = []

            app_reviews, _ = reviews(app_id = app_id, lang = "en", count = 50, sort = Sort.NEWEST)

            for app_review in app_reviews:
                app_review['appId'] = app_id

            all_apps_reviews.extend(app_reviews)
        
        FileHandler.save_data_to_json(all_apps_reviews, filename = Config.REVIEWS_DATA_FILENAME)
        return all_apps_reviews