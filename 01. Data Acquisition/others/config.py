class Config:
    APPS_DATA_FILENAME = "./data/apps_data.json"
    REVIEWS_DATA_FILENAME = "./data/reviews_data.json"
    APPS_ID_FILENAME  = "./data/app_ids.txt"
    APPS_IMAGE_CAPTIONS_FILENAME  = "./data/app_image_captions.csv"

    COUNTRIES_LIST = ['us', 'id', 'sg', 'my', 'ph']
    LANGUAGE_LIST = ['en', 'id']

    KEYWORDS_LIST = [
        # general
        "education", "learning", "course", "teach", "online course", "homework helper",  "distance learning", "micro learning", "brain training", "exam", "exam preparation", "pedagogical", "e-learning"
        
        # STEM
        "science education", "coding", "coding for kids", "preschool games", "dictionary", "study-aids", "math", "anatomy", "ethical hacking", "STEM", "robotics", "programming", "computer science", "artificial intelligence", "machine learning", "data science", "cybersecurity", "space exploration", "environmental science", "3D modeling", "virtual labs", 
        
        # language
        "language learning", "learn english", "grammar practice", "flashcards",  "vocabulary", "ESL", "IELTS", "TOEFL", "TOEIC", "reading practice", "listening practice", "speaking practice",
        
        # academic subjects
        "school subjects", "academic learning", "calculus", "linear algebra", "discrete math", "organic chemistry", "quantum physics", "anatomy physiology", "macroeconomics", "statistics", "periodic table", "medical dictionary", "engineering formulas", "science", "biology", "chemistry", "accounting", "finance", "business studies", "medical dictionary", "medical education", "nursing exam", "medical exam prep",

        # kids edu
        "kids learning", "learning for kids", "kids education", "preschool", "preschool learning", "kindergarten","early learning", "early education","phonics", "sight words", "kids games", "educational games", "ABC learning", "alphabet learning","kids math", "kids reading","montessori",

        # exam & certif prep
        "exam prep", "competitive exams", "college entrance exam", "SAT", "ACT", "GRE", "GMAT", "MCAT", "USMLE", "civil service exam", "certification", "professional certification",

        # skill
        "skill development", "skill learning","career learning", "professional learning", "soft skills","communication skills", "critical thinking", "problem solving", "personal development",
    ]

    EXCLUDE_GENRE = [
        "Music & Audio", "Casual", "Card", "Lifestyle", "Parenting", "Health & Fitness", "Puzzle", "Travel & Local", "Finance", "Social", "Tools", "Action", "Entertainment", "Art & Design", "Personalization", "Role Playing", "Simulation", "Events", "Puzzle", "Communication", "Arcade", "Business", "Food & Drink", "Photography", "Maps & Navigation", "Music", "Sports", "Board", "Weather", "House & Home", "Racing", "Strategy", "Adventure", "Shopping", "Video Players & Editors", "Comics",  "Productivity", "Auto & Vehicles", "Dating", "Casino", "Beauty"
        ]

    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 150

    INDEX_NAME = "pedagogical-google-play-apps"
    APP_INFO_NAMESPACE = "app-info"
    APP_REVIEWS_NAMESPACE = "reviews"
    APPS_IMAGE_CAPTIONS_NAMESPACE = "image-captions"