import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Скачиваем стоп-слова (выполнить один раз)
nltk.download('stopwords')
nltk.download('punkt')


def preprocess_text(text):
    if isinstance(text, float):
        return ""

    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление ссылок
    text = re.sub(r'http\S+', '', text)

    # Удаление упоминаний пользователей
    text = re.sub(r'@\w+', '', text)

    # Удаление специальных символов и цифр
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Токенизация и удаление стоп-слов
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

    return ' '.join(tokens)
