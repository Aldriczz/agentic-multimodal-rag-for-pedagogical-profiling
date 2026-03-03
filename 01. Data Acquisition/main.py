from apps_scraper import AppsScraper
from reviews_scraper import ReviewsScraper
from ocr import VisualCaptioning
from ingestion import Ingestion
from others.utils import FileHandler
from others.config import Config

def __main__():
    # Step 1: Scrape apps data
    print(Config.KEYWORDS_LIST)
    apps = AppsScraper.get_apps(Config.KEYWORDS_LIST)
    # apps = FileHandler.load_data_from_json(filename = Config.APPS_DATA_FILENAME)

    # Step 2: Scrape reviews data
    # reviews = ReviewsScraper.get_reviews_by_app_id(apps)

    # Step 3: Generate visual captions
    # ocr_captions = VisualCaptioning.generate_caption_from_apps_screenshots(apps)

    # Step 4: Upsert data to Pinecone
    # Ingestion.upsert_app_info(apps)



def tes():
    index = Ingestion.get_index()
    index.delete(delete_all = True, namespace = Config.APP_INFO_NAMESPACE)

if __name__ == "__main__":
    __main__()
    # tes()

