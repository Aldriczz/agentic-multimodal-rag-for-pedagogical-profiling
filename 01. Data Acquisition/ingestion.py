import os
import time

from dotenv import load_dotenv
from preprocess import Preprocessor
from pinecone import Pinecone
from others.config import Config
from tqdm import tqdm

class Ingestion:
    def get_index():
        load_dotenv()
        pc = Pinecone(api_key=os.getenv("PINECONE_TES"))
        index = pc.Index(name=Config.INDEX_NAME)
        return index

    def upsert_app_info(apps : list):
        records = []

        for app in apps:
            chunks = Preprocessor.preprocess_app_info(app)
            for i, chunk in enumerate(chunks):
                records.append({
                    "id": f"{app['appId']}__info__{i}",
                    "appName": app["title"],
                    "appId": app["appId"],
                    "content": chunk,
                    "developer": app["developer"] or "",
                    "released": app["released"] or "",
                    "lastUpdatedOn": app["lastUpdatedOn"] or "",
                    "updated": app["updated"] or "",
                    "free": app["free"] or False,
                    "price": app["price"] or 0,
                    "currency": app["currency"] or "",
                    "realInstalls": app["realInstalls"] or 0,
                    "score": app["score"] or 0,
                    "ratingsNumber": app["ratings"] or 0,
                    "reviewsNumber": app["reviews"] or 0,
                    "contentRating": app["contentRating"] or "",
                })
        Ingestion.batch_upsert(Ingestion.get_index(), Config.APP_INFO_NAMESPACE, records)

    def upsert_app_reviews(reviews : list):
        records = []

        for i, review in enumerate(reviews):
            chunks = Preprocessor.preprocess_app_reviews(review)
            for j, chunk in enumerate(chunks):
                short_reviews = []
                short_scores = []

                chunk_slices = chunk.split(" ")

                if len(chunk_slices) < 5:
                    short_reviews.append(chunk)
                    short_scores.append(review["score"])
                    continue

                records.append({
                    "id": f"{review['appId']}__reviews__{i}_{j}",
                    "appId": review["appId"],
                    "content": chunk,
                    "score": review["score"] or 0,
                    "date": review["at"] or "", 
                })

        if short_reviews:
            aggregated_short_reviews = " | ".join(short_reviews)
            print(f"Short reviews: {aggregated_short_reviews}")
            records.append({
                    "id": f"{review['appId']}__reviews__aggregated_short",
                    "appId": review["appId"],
                    "content": aggregated_short_reviews,
                    "score": sum(short_scores) / len(short_scores) if short_scores else 0,
                    "date": str(time.time()) or "", 
                })
        Ingestion.batch_upsert(Ingestion.get_index(), Config.APP_REVIEWS_NAMESPACE, records)

    def upsert_app_ocr_caption(ocr_captions : list):
        records = []

        for ocr in ocr_captions:
            text = Preprocessor.preprocess_app_ocr_captions(ocr)
            if len(text) == 0:
                continue
            records.append({
                "id": ocr["caption_id"],
                "appId": ocr["app_id"],
                "content": text,
                "screenshotUrl": ocr["screenshot"] or "",
            })
        
        Ingestion.batch_upsert(Ingestion.get_index(), Config.APPS_OCR_CAPTIONS_NAMESPACE, records)

    def batch_upsert(index, namespace : str, records : list, batch_size : int = 96):
        count = 0
        for i in tqdm(range(0, len(records), batch_size)):  
            batch = records[i:i + batch_size]
            index.upsert_records(
                namespace=namespace,
                records=batch
            )
            count += 1
            if count % 15 == 0:
                time.sleep(60)

        print(f"Upserted {len(records)} records to namespace {namespace}")

    def upsert_all(apps : list, reviews : list, ocr_captions : list):
        print("Ingesting app info. . . . . .")
        Ingestion.upsert_app_info(apps)
        print("Ingesting app reviews. . . . . .")
        Ingestion.upsert_app_reviews(reviews)
        print("Ingesting app ocr captions. . . . . .")
        Ingestion.upsert_app_ocr_caption(ocr_captions)