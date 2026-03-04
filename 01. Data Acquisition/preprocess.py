from email.mime import text
import re
import emoji
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from others.config import Config


class Preprocessor:
    def clean_text(text : str, remove_emoji : bool = False) -> str:
        # remove html tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # remove emojis
        if remove_emoji:
            text = emoji.replace_emoji(text, replace='')
        else:
            text = emoji.demojize(text)

        text = re.sub(r'\"', '', text).strip()

        text = re.sub(r'\r+', '', text).strip()

        text = str.replace(text, '\xa0', '').strip()

        text = str.replace(text, "\'", "").strip()

        text = str.replace(text, '"', '').strip()

        text = re.sub(r'\s+', ' ', text).strip()

        # standardize points
        text = re.sub(r'[●•·❖★✔>*]', '-', text)
        
        return text
    
    def enrich_chunk(name : str, summary : str, description : str) -> str:
        return f"Application Name: {name} Summary: {summary} Description: {description}"
    
    def chunking(text : str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = Config.CHUNK_SIZE,     
            chunk_overlap  = Config.CHUNK_OVERLAP,  
            separators=["\n\n", "\n", ". ", " "] 
        )

        chunks = text_splitter.split_text(text)
        return chunks
    
    def preprocess_app_info(app_info: dict):
        name = app_info.get("title", "")
        summary = app_info.get("summary", "")
        description = app_info.get("description", "")

        text = Preprocessor.enrich_chunk(name, summary, description)
        text = Preprocessor.clean_text(str(text), remove_emoji = True)

        chunks = Preprocessor.chunking(text)

        return chunks

    def preprocess_app_reviews(review: dict):
        content = review.get("content", "")

        text = Preprocessor.clean_text(str(content), remove_emoji = False)

        chunks = Preprocessor.chunking(text)

        return chunks

    def preprocess_app_ocr_captions(review: dict):
        content = review.get("caption", "")

        text = Preprocessor.clean_text(str(content), remove_emoji = True)

        chunks = Preprocessor.chunking(text)

        return chunks