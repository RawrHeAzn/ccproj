import pandas as pd
from sqlalchemy import create_engine
import urllib
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# from sklearn.preprocessing import LabelEncoder # Not needed for this approach
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Database Connection Setup (Same as main.py) ---
server = 'retail-sql-server22.database.windows.net'
database = 'RetailDB22'
username = 'retailadmin22'
password = 'Datalord22'
driver= '{ODBC Driver 17 for SQL Server}'

params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)

# --- Configuration ---
# <<< CHANGE THIS to the commodity you want to predict >>>
TARGET_COMMODITY = 'DAIRY' 
# <<< Output path for the trained model >>>
MODEL_OUTPUT_PATH = 'models/basket_rf_dairy_model.pkl'
# <<< --- >>>
TEST_SIZE = 0.3
RANDOM_STATE = 42

def fetch_data(conn):
    """Grabs the basket number and commodity name from transactions+products tables"""
    logger.info("Fetching transaction data...")
    query = """
        SELECT 
            t.BASKET_NUM, 
            p.COMMODITY 
        FROM transactions t
        JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
        WHERE p.COMMODITY IS NOT NULL AND p.COMMODITY <> '' -- skip rows with no commodity name
    """
    df = pd.read_sql(query, conn)
    logger.info(f"Got {len(df)} transaction lines.")
    return df

def preprocess_data(df, target_commodity):
    """Turns the raw transaction data into a format the model can understand"""
    logger.info("Preprocessing data...")
    
    # First, make sure the target commodity actually exists in our data!
    if target_commodity not in df['COMMODITY'].unique():
        valid_commodities = df['COMMODITY'].unique()
        logger.error(f"Oops! Target commodity '{target_commodity}' wasn't found.")
        logger.error(f"Available ones look like: {list(valid_commodities[:100])}...") # Show some examples
        raise ValueError(f"Target commodity '{target_commodity}' is not in the data.")

    # Create the basket matrix using pivot_table
    # index=basket, columns=commodity, values=1 (if item exists), fill missing with 0
    logger.info("Making the basket matrix (this might take a sec)...")
    basket_pivot = df.pivot_table(index='BASKET_NUM', columns='COMMODITY', aggfunc=lambda x: 1, fill_value=0)
    
    # crosstab could also work but pivot_table is maybe safer?
    # basket_matrix = pd.crosstab(df['BASKET_NUM'], df['COMMODITY'])
    # basket_matrix = (basket_matrix > 0).astype(int)

    logger.info(f"Created basket matrix! Shape: {basket_pivot.shape} (baskets, commodities)")

    # Split into features (X) and target (y)
    # y is the column for our target commodity
    # X is all OTHER commodity columns
    if target_commodity not in basket_pivot.columns:
         # This should be caught earlier, but better safe than sorry
         raise ValueError(f"Target column '{target_commodity}' disappeared somehow? Uh oh.")
         
    y = basket_pivot[target_commodity]
    X = basket_pivot.drop(columns=[target_commodity])
    
    logger.info(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
    
    # Check if the target variable is all 0s or all 1s - model can't learn then
    if len(y.unique()) < 2:
         logger.error(f"Target '{target_commodity}' only has one value ({y.unique()[0]})! Maybe everyone buys it, or nobody? Can't train.")
         raise ValueError(f"Target variable '{target_commodity}' is constant, useless for training.")
         
    # How often is the target item present vs absent? (Good to know if it's imbalanced)
    logger.info(f"Target '{target_commodity}' distribution (0=Absent, 1=Present):\n{y.value_counts(normalize=True)}")
    
    return X, y # Return features and the target column

def train_evaluate_model(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Trains the Random Forest model and sees how well it does"""
    logger.info("Splitting data for training/testing...")
    # stratify=y tries to keep the same percentage of 0s and 1s in both train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y) 

    logger.info(f"Training the Random Forest model to predict: '{y.name}'...")
    # class_weight='balanced' helps if one class (e.g., target present) is much rarer
    # n_jobs=-1 uses all CPU cores to speed it up
    model = RandomForestClassifier(n_estimators=100, 
                                 random_state=random_state, 
                                 class_weight='balanced', 
                                 n_jobs=-1)
    model.fit(X_train, y_train)

    logger.info("Okay, model trained. Let's see how it did on the test data...")
    y_pred = model.predict(X_test) # The model's predictions (0 or 1)
    y_proba = model.predict_proba(X_test)[:, 1] # The model's confidence (probability) for class 1
    
    accuracy = accuracy_score(y_test, y_pred)
    # Classification report gives precision, recall, f1-score - useful for imbalanced data
    report = classification_report(y_test, y_pred, target_names=['Absent', 'Present'])
    
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info(f"Detailed Report:\n{report}")
    
    # Which other commodities were most important for predicting the target?
    try:
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        logger.info("\nTop 10 Most Important Features (Commodities):")
        logger.info(feature_importances.nlargest(10))
    except Exception as e:
        logger.warning(f"Couldn't get feature importances: {e}")

    return model # Return the trained model object

def save_model(model, features, filepath):
    """Saves the trained model AND the list of feature names it uses"""
    logger.info(f"Saving model & features to {filepath}..." )
    try:
        # Make sure the directory exists (e.g., 'models/')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Put model and feature list into a dictionary
        model_data = {
            'model': model,
            'features': features.tolist() # Need to convert pandas Index to a list
        }
        
        # Save the dictionary using joblib
        joblib.dump(model_data, filepath)
        logger.info(f"Model and features saved ok! Path: {filepath}")
    except Exception as e:
        logger.error(f"Uh oh, couldn't save the model: {e}")


if __name__ == "__main__":
    conn = None
    logger.info("--- Starting Basket Prediction Model Training Script ---")
    try:
        logger.info("Connecting to the database...")
        conn = engine.connect()
        logger.info("DB connection successful.")
        
        # 1. Get the data
        df_transactions = fetch_data(conn)
        
        if not df_transactions.empty:
            # 2. Process the data
            X_features, y_target = preprocess_data(df_transactions, TARGET_COMMODITY)
            
            # 3. Train the model
            trained_classifier = train_evaluate_model(X_features, y_target)
            
            # 4. Save the model (and its features)
            save_model(trained_classifier, X_features.columns, MODEL_OUTPUT_PATH)
        else:
            logger.warning("Didn't get any data from DB. Skipping training.")
            
    except ValueError as ve:
         # Handle specific errors we defined (like bad target commodity)
         logger.error(f"Problem with setup or data: {ve}")
    except Exception as e:
        # Catch any other unexpected problems
        logger.error(f"Something unexpected went wrong: {e}", exc_info=True)
    finally:
        # Always make sure to close the DB connection
        if conn:
            conn.close()
            logger.info("Database connection closed.")
        logger.info("--- Training script finished. ---") 