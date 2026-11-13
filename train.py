import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from load_data import load_data
from features import build_corpus, get_tfidf


def get_split(test_size=0.2, seed=42):
    df = load_data()
    X = build_corpus(df)
    y = df["rating"]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def main():
    X_train, X_val, y_train, y_val = get_split()

    vec = get_tfidf()
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)

    model = MultinomialNB(alpha=1.0)
    model.fit(Xtr, y_train)

    print("Validation accuracy:", round(model.score(Xva, y_val), 4))
    joblib.dump((vec, model), "model.joblib")
    print("Saved model.joblib")


if __name__ == "__main__":
    main()
