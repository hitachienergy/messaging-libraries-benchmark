import pandas as pd
import joblib
import argparse
import os

from bml_optimizer.utils.logger import get_logger
from bml_optimizer.simulator.simulator import BML, parse_bml, Protocol

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
FEATURES = ["Messages Sent", "Payload Length", "Pub Delay", "Pub Interval", "Num Subscribers"]
logger = get_logger(__name__)


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV file into a pandas DataFrame
    """
    if not os.path.isfile(csv_path): 
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def transform_csv(df: pd.DataFrame, fom: str) -> pd.DataFrame:
    """
    Group by the specified columns and compute Best Library as the library with minimum FOM per group.
    Use aggregation to compute minimum latency for each group then merge to obtain the associated "Library"
    """
    group_cols = ["Protocol"] + FEATURES
    if fom == "Combined":
        ##
        # Have the possibility of having "Combined" instead, and produce a column "Combined" in the csv: (Throughput / Avg Latency)
        ## 
        df["Combined"] = df["Payload Throughput"] / df["Avg Latency"]
        metric_col = "Combined"

    else:
        metric_col = fom

    def get_best_library(group):
        if fom == "Avg Latency": best_idx = group[metric_col].idxmin()
        elif fom == "Payload Throughput" or fom == "Throughput" or fom == "Combined": best_idx = group[metric_col].idxmax()
        else:
            raise ValueError(f"Invalid Figure of Merit: {fom}")

        best = group.loc[best_idx, "Library"]
        group = group.copy()[group_cols]
        group["Best Library"] = best
        return group.head(1)

    transformed_df = df.groupby(group_cols, as_index=False).apply(get_best_library).reset_index(drop=True)
    logger.debug(f"Transformed DataFrame:\n{transformed_df.head()}")
    return transformed_df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Preprocess the data for training the classifier
        Encoding categorical data for "Protocol" and "Best Library" columns
    """
    X = df[FEATURES].copy()  # avoid SettingWithCopyWarning
    y = df["Best Library"]

    le_library = LabelEncoder()
    y = le_library.fit_transform(y)

    return X, y, le_library


def train_model(protocol: str, X: pd.DataFrame, y: pd.Series, dump_folder: str) -> RandomForestClassifier:
    """
    Train the model
        Splitting the data into training and test sets
        Initializing and training the classifier
        Evaluating the model
    """
    logger.info(f"Training model with features: {list(X.columns)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    logger.info(f"Training data size: {len(X_train)}, Test data size: {len(X_test)}")

    param_grid = {
        'n_estimators': [100, 1000],
        'max_depth': [None, 10, 100],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=RANDOM_STATE),
                               param_grid=param_grid,
                               cv=3,
                               n_jobs=-1,
                               verbose=2)
    grid_search.fit(X_train, y_train)

    clf = grid_search.best_estimator_
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Classification Report:\n{report}")

    if not os.path.exists(dump_folder): os.makedirs(dump_folder)
    log_file_path = os.path.join(dump_folder, f"training_log_{protocol}.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Accuracy: {accuracy}\n")
        log_file.write(f"Classification Report:\n{report}\n")

    return clf


def dump_model(protocol: str, 
               dump_folder: str, 
               model: RandomForestClassifier, 
               library_encoder: LabelEncoder) -> None:
    """
    Dump the trained model using joblib
    Each model is specific to a protocol, and is thus dumped as classifier_{protocol}.pkl and encoders_{protocol}.pkl
    """
    logger.info(f"Saving model to folder: {dump_folder}")
    if not os.path.exists(dump_folder): os.makedirs(dump_folder)
    classifier_path = os.path.join(dump_folder, f"classifier_{protocol}.pkl")
    encoders_path = os.path.join(dump_folder, f"encoders_{protocol}.pkl")

    joblib.dump(model, classifier_path)
    joblib.dump({"library": library_encoder}, encoders_path)
    logger.info(f"Model and encoders dumped to: {dump_folder}")


def predict_best_library(dump_folder: str,
                         protocol: Protocol,
                         messages_sent: int,
                         payload_length: int,
                         pub_delay: float,
                         pub_interval: float,
                         num_subscribers: int) -> BML:
    """
    Load a dumped model and predict the best library for the given input parameters.
    Using the model corresponding to the specified protocol
    """
    protocol_str: str = protocol.name.lower()

    model_path = os.path.join(dump_folder, f"classifier_{protocol_str}.pkl")
    encoders_path = os.path.join(dump_folder, f"encoders_{protocol_str}.pkl")

    if not os.path.exists(model_path) or not os.path.exists(encoders_path):
        raise FileNotFoundError(f"Model or encoders not found for protocol '{protocol_str}' in folder '{dump_folder}'")

    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)

    library_encoder = encoders["library"]

    input_df = pd.DataFrame({
        "Messages Sent": [messages_sent],
        "Payload Length": [payload_length],
        "Pub Delay": [pub_delay],
        "Pub Interval": [pub_interval],
        "Num Subscribers": [num_subscribers]
    })

    prediction = model.predict(input_df)
    predicted_library: str = library_encoder.inverse_transform(prediction)[0]

    logger.debug(f"Predicted Best Library: {predicted_library}")
    return parse_bml(predicted_library)


def main():
    """
    For each protocol (inproc, ipc, tcp), train a classifier and dump the model
    """
    parser = argparse.ArgumentParser(description="Train a classifier with given CSV and figure of merit")
    parser.add_argument("--fom", type=str, default="Avg Latency", help="Figure of Merit column name")
    parser.add_argument("--input_csv", type=str, default="results/results.csv", help="Path to the input CSV file")
    parser.add_argument("--dump_folder", type=str, default="model", help="Output path for the dumped model")
    args = parser.parse_args()

    df: pd.DataFrame = load_csv_data(args.input_csv)
    logger.info(f"Loaded CSV from {args.input_csv} with {len(df)} rows")
    logger.info(f"FOM chosen: {args.fom}")
    df_transformed: pd.DataFrame = transform_csv(df, args.fom)

    unique_protocols = df_transformed["Protocol"].unique()
    for protocol in unique_protocols:
        protocol_df = df_transformed[df_transformed["Protocol"] == protocol]
        logger.info(f"Protocol '{protocol}' DataFrame size: {len(protocol_df)} rows")
        X, y, library_encoder = preprocess_data(protocol_df)
        classifier = train_model(protocol.lower(), X, y, dump_folder=args.dump_folder)
        dump_model(protocol.lower(), args.dump_folder, classifier, library_encoder)

if __name__ == "__main__":
    main()