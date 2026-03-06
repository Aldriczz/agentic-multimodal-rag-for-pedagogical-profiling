from apps_scraper import AppsScraper
from reviews_scraper import ReviewsScraper
from ocr import VisualCaptioning
from ingestion import Ingestion
from others.utils import FileHandler
from others.config import Config

def __main__():
    # Step 1: Scrape apps data
    apps = AppsScraper.get_apps(Config.KEYWORDS_LIST)

    # Step 2: Scrape reviews data
    reviews = ReviewsScraper.get_reviews_by_app_id(apps)

    # Step 3: Generate visual captions
    ocr_captions = VisualCaptioning.generate_caption_from_apps_screenshots(apps)

    # (load datas)
    # apps, reviews, ocr_captions = load_datas()

    # Step 4: Upsert data to Pinecone
    Ingestion.upsert_all(apps, reviews, ocr_captions)


def load_datas():
    apps = FileHandler.load_data_from_json(filename = Config.APPS_DATA_FILENAME)
    reviews = FileHandler.load_data_from_json(filename = Config.REVIEWS_DATA_FILENAME)
    ocr_captions = FileHandler.load_data_from_csv(filename = Config.APPS_OCR_CAPTIONS_FILENAME)
    
    return apps, reviews, ocr_captions

def delete_namespace():
    index = Ingestion.get_index()
    index.delete(delete_all = True, namespace = Config.APPS_OCR_CAPTIONS_NAMESPACE)

if __name__ == "__main__":
    __main__()
    # delete_namespace()

