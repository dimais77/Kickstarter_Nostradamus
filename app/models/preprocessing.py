from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

class ModelTrainer:
    def __init__(self, classifier=LogisticRegression(max_iter=1000)):
        self.wnl = WordNetLemmatizer()
        self.classifier = classifier
        self.numeric_features = ["goal", "campaign_duration", "started_month"]
        self.categorical_features = ["currency", "category_subcategory"]
        self.text_features = ["blurb"]
        self.preprocessor = self._build_preprocessor()

    def _preprocessing(self, line):
        line = line.lower()
        line = re.sub(r'[^\w\s]', ' ', line)
        line = ' '.join([self.wnl.lemmatize(token) for token in line.split()])
        return line

    def _build_preprocessor(self):
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        text_transformer = Pipeline(steps=[
            ('tfidf', TfidfVectorizer(
                stop_words='english',
                preprocessor=self._preprocessing,
                max_df=0.7,
                min_df=5
            ))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, self.numeric_features),
            ("cat", categorical_transformer, self.categorical_features),
            ("text", text_transformer, "blurb")
        ])

        return preprocessor

    def train(self, X, y):
        clf = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("classifier", self.classifier)
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        clf.fit(X_train, y_train)
        return clf, X_test, y_test