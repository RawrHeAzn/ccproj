# backend/main.py

from fastapi import Request, FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool # handy for running blocking stuff like model.predict
import joblib 
from joblib import load 
import pyodbc 
import pandas as pd
import io  # needed for file uploads
from pydantic import BaseModel # helps define what our API expects
from passlib.context import CryptContext # for hashing passwords, security stuff!
import logging # good for seeing what's going on / debugging
from sqlalchemy import create_engine, text # Import text
import urllib # used for formatting the db connection string
from apscheduler.schedulers.background import BackgroundScheduler # runs tasks on a schedule, neat!
from contextlib import asynccontextmanager # for startup/shutdown events
# Apriori stuff for finding item pairs
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import asyncio # For triggering background tasks

# Setup logging so we can see messages in the console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global State for Update Tracking ---
dashboard_update_in_progress = False

# --- Load up our trained models --- 
# Load the Customer Lifetime Value model first
clv_model = joblib.load('models/clv_model.pkl')

# Now try loading the basket prediction model (might fail if training wasn't run)
basket_model_data = None
basket_model = None
basket_features = None
BASKET_MODEL_PATH = 'models/basket_rf_dairy_model.pkl' # where did we save this again?
TARGET_ITEM_NAME = 'DAIRY' # what item is this model trying to predict? Dairy!

try:
    basket_model_data = joblib.load(BASKET_MODEL_PATH)
    basket_model = basket_model_data['model'] # the actual classifier
    basket_features = basket_model_data['features'] # the commodity names it knows
    logger.info(f"Loaded the basket prediction model! It knows {len(basket_features)} features.")
except FileNotFoundError:
    logger.error(f"Couldn't find the basket model at {BASKET_MODEL_PATH}. The prediction part won't work!")
    basket_model_data = None # make sure it's None so we know it failed
except Exception as e:
    logger.error(f"Some other error loading the basket model: {e}")
    basket_model_data = None


# --- App startup/shutdown stuff --- 
# This runs when the app starts and stops
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run this when FastAPI starts up
    logger.info("App starting up... let's load the dashboard data first.")
    await update_dashboard_data() # run the data update task
    logger.info("Initial dashboard data loaded. Ready to go!")
    
    yield # the app runs while this is yielded
    
    # Run this when FastAPI shuts down
    logger.info("App shutting down... stopping the scheduled task.")
    scheduler.shutdown()
    logger.info("Scheduler stopped.")

# Create the FastAPI app and tell it to use our lifespan thingy
app = FastAPI(lifespan=lifespan)

# --- Database Connection Details --- (TODO: maybe use environment variables later?)
server = 'retail-sql-server22.database.windows.net'
database = 'RetailDB22'
username = 'retailadmin22'
password = 'Datalord22' # lol
driver= '{ODBC Driver 18 for SQL Server}'

# --- SQLAlchemy Setup --- 
# This makes the connection string look right for SQLAlchemy
params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    f"Encrypt=yes;"                # 🔥 Must add this
    f"TrustServerCertificate=no;"  # 🔥 And this (no skipping certs)
    f"Connection Timeout=30;"       # 🔥 Optional good practice
)
# The SQLAlchemy engine - needed for pandas read_sql
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)

# Function to get a SQLAlchemy connection (used by background tasks & pandas)
def get_sqlalchemy_connection(): 
    # Returns a connection object from the engine
    try:
        conn = engine.connect()
        return conn
    except Exception as ex:
        logger.error(f"SQLAlchemy connection failed :( Error: {ex}")
        # Don't kill the app here, maybe the background task can recover?
        return None # return None if it breaks

# --- PyODBC Setup --- (older way, maybe needed for some specific stuff?)
conn_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;'

# Function to get a plain pyodbc connection
def get_pyodbc_connection(): 
    # Returns a raw pyodbc connection object
    try:
        conn = pyodbc.connect(conn_str, autocommit=False) # turn autocommit off just in case
        return conn
    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        logger.error(f"pyodbc connection failed :( Error: {sqlstate} - {ex}")
        raise HTTPException(status_code=500, detail="Database connection broke")

# --- Dashboard Data Calculation Functions --- 
# These run in the background to get data ready for the dashboard charts

# Top 10 spenders
def _fetch_top_spenders(conn):
    query = '''
        SELECT TOP 10 HSHD_NUM, SUM(SPEND) AS total_spend
        FROM transactions
        GROUP BY HSHD_NUM
        ORDER BY total_spend DESC;
    '''
    df = pd.read_sql(query, conn)
    df = rename_columns(df, {'HSHD_NUM': 'Hshd_num'}) # match frontend naming
    return df.to_dict(orient='records')

# Loyalty trends (spend over time by loyal/not loyal)
def _fetch_loyalty_trends(conn):
    query = '''
        SELECT h.LOYALTY_FLAG, t.YEAR, t.WEEK_NUM, SUM(t.SPEND) AS total_spend
        FROM transactions t
        JOIN households h ON t.HSHD_NUM = h.HSHD_NUM
        GROUP BY h.LOYALTY_FLAG, t.YEAR, t.WEEK_NUM
        ORDER BY t.YEAR, t.WEEK_NUM, h.LOYALTY_FLAG;
    '''
    df = pd.read_sql(query, conn)
    df = rename_columns(df, { # match frontend naming
        'LOYALTY_FLAG': 'Loyalty_flag',
        'YEAR': 'Year',
        'WEEK_NUM': 'Week_num'
    })
    return df.to_dict(orient='records')

# Avg spend by income group
def _fetch_engagement_by_income(conn):
    query = '''
        SELECT
            h.INCOME_RANGE AS income_bracket,
            AVG(t.SPEND) AS avg_spend
        FROM transactions t
        JOIN households h ON t.HSHD_NUM = h.HSHD_NUM
        WHERE h.INCOME_RANGE IS NOT NULL
        GROUP BY h.INCOME_RANGE
        -- ORDER BY h.INCOME_RANGE; -- ordering alphabetically is WRONG!
    '''
    df = pd.read_sql(query, conn)
    
    # Need to sort these income ranges manually because '100k' comes before '50k' alphabetically
    income_order = [
        '<25K', 
        '25-34K',
        '35-49K',
        '50-74K', 
        '75-99K', 
        '100-149K', 
        '150K+' # Double-check these strings match the DB exactly!
        # Add any others if they exist
    ]
    
    # Tell pandas this column has a specific order
    df['income_bracket'] = pd.Categorical(df['income_bracket'], categories=income_order, ordered=True)
    
    # Get rid of any rows that didn't match our defined order (just in case)
    df = df.dropna(subset=['income_bracket'])
    
    # Now sort it correctly!
    df = df.sort_values('income_bracket')
    
    return df.to_dict(orient='records')

# Spend split by private label vs national brand
def _fetch_brand_preference_split(conn):
    query = '''
        SELECT
            p.BRAND_TY AS brand_type,
            SUM(t.SPEND) AS total_spend
        FROM transactions t
        JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
        WHERE p.BRAND_TY IS NOT NULL
        GROUP BY p.BRAND_TY
        ORDER BY total_spend DESC;
    '''
    df = pd.read_sql(query, conn)
    return df.to_dict(orient='records')

# Top 10 pairs of commodities bought together
def _fetch_frequent_pairs(conn):
    # This SQL is a bit complex, finds pairs within the same basket
    query = '''
        WITH BasketPairs AS (
            SELECT DISTINCT
                t1.BASKET_NUM,
                LEAST(t1.PRODUCT_NUM, t2.PRODUCT_NUM) AS Product1,
                GREATEST(t1.PRODUCT_NUM, t2.PRODUCT_NUM) AS Product2
            FROM transactions t1
            JOIN transactions t2 ON t1.BASKET_NUM = t2.BASKET_NUM AND t1.PRODUCT_NUM < t2.PRODUCT_NUM
        )
        SELECT TOP 10
            p1.COMMODITY AS item1,
            p2.COMMODITY AS item2,
            COUNT(*) as count
        FROM BasketPairs bp
        JOIN products p1 ON bp.Product1 = p1.PRODUCT_NUM
        JOIN products p2 ON bp.Product2 = p2.PRODUCT_NUM
        GROUP BY p1.COMMODITY, p2.COMMODITY
        HAVING p1.COMMODITY <> p2.COMMODITY -- Make sure item1 and item2 are different
        ORDER BY count DESC;
    '''
    df = pd.read_sql(query, conn)
    return df.to_dict(orient='records')

# Top 10 most popular commodities by total spend
def _fetch_popular_products(conn):
    query = '''
        SELECT TOP 10
            p.COMMODITY AS commodity,
            SUM(t.SPEND) AS total_spend
        FROM transactions t
        JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
        GROUP BY p.COMMODITY
        ORDER BY total_spend DESC;
    '''
    df = pd.read_sql(query, conn)
    return df.to_dict(orient='records')

# Total spend aggregated by month/year
def _fetch_seasonal_trends(conn):
    query = '''
        SELECT
            YEAR(t.DATE) AS year,
            MONTH(t.DATE) AS month,
            SUM(t.SPEND) AS total_spend
        FROM transactions t
        GROUP BY YEAR(t.DATE), MONTH(t.DATE)
        ORDER BY year, month;
    '''
    df = pd.read_sql(query, conn)
    return df.to_dict(orient='records')

# Find customers who haven't bought anything in 8 weeks (potential churn)
def _fetch_churn_risk(conn):
    # Find the latest date in the whole dataset first
    max_date_query = "SELECT MAX(DATE) FROM transactions"
    max_date_df = pd.read_sql(max_date_query, conn)
    if max_date_df.empty or max_date_df.iloc[0, 0] is None:
        logger.warning("Can't find the max date, churn analysis won't work.")
        return {"at_risk_list": [], "summary_stats": {}} # Return empty stuff
    max_date = pd.to_datetime(max_date_df.iloc[0, 0])
    # Cutoff date is 8 weeks before the last transaction date
    cutoff_date = max_date - pd.Timedelta(weeks=8)
    
    logger.info(f"Churn Check: Max date={max_date}, Cutoff={cutoff_date}")

    # Find households whose latest purchase was BEFORE the cutoff
    query = f"""
        SELECT 
            h.HSHD_NUM, 
            MAX(t.DATE) AS last_purchase_date,
            h.LOYALTY_FLAG,
            h.INCOME_RANGE,
            h.HSHD_SIZE, -- fixed this column name earlier
            h.CHILDREN
        FROM households h
        JOIN transactions t ON h.HSHD_NUM = t.HSHD_NUM
        GROUP BY h.HSHD_NUM, h.LOYALTY_FLAG, h.INCOME_RANGE, h.HSHD_SIZE, h.CHILDREN -- group by all household fields
        HAVING MAX(t.DATE) < ? -- the important condition!
        ORDER BY last_purchase_date ASC; -- show oldest first
    """
    try:
        # Need to pass the date parameter correctly for the SQL query
        df = pd.read_sql(query, conn, params=[(cutoff_date.strftime('%Y-%m-%d'),)])
    except Exception as e:
        logger.error(f"DB error fetching churn risk data: {e}")
        return {"at_risk_list": [], "summary_stats": {"error": "Failed to fetch data"}}

    if df.empty:
        logger.info("No customers seem to be at risk of churning. Good!")
        return {"at_risk_list": [], "summary_stats": {"count_by_loyalty": [], "count_by_income": []}}

    # Make column names nice for the frontend
    df = rename_columns(df, {
        'HSHD_NUM': 'Hshd_num',
        'last_purchase_date': 'LastPurchaseDate',
        'LOYALTY_FLAG': 'Loyalty_flag',
        'INCOME_RANGE': 'IncomeRange',
        'HSHD_SIZE': 'HshdSize',
        'CHILDREN': 'Children'
    })
    # Dates need to be strings for JSON
    df['LastPurchaseDate'] = pd.to_datetime(df['LastPurchaseDate']).dt.strftime('%Y-%m-%d')
    # Replace any weird NaN values with None (null in JSON)
    df = df.astype(object).where(pd.notnull(df), None)
    
    # --- Calculate some summary stats about the at-risk people --- 
    summary_stats = {}
    try:
        # How many loyal vs non-loyal are at risk?
        loyalty_counts = df['Loyalty_flag'].value_counts().reset_index()
        loyalty_counts.columns = ['loyalty_flag', 'count']
        summary_stats['count_by_loyalty'] = loyalty_counts.to_dict(orient='records')

        # How many in each income group are at risk?
        income_counts = df['IncomeRange'].value_counts().reset_index()
        income_counts.columns = ['income_range', 'count']
        summary_stats['count_by_income'] = income_counts.to_dict(orient='records')
        
        # Could add more summaries here maybe?
        
    except Exception as e:
        logger.error(f"Couldn't calculate churn summary stats: {e}")
        summary_stats['error'] = "Failed generating summaries"

    logger.info(f"Found {len(df)} customers potentially churnin'!")
    
    # Send back the list of customers and the summaries
    return {
        "at_risk_list": df.to_dict(orient='records'),
        "summary_stats": summary_stats
    }

# --- Apriori Function --- (Finds which items are often bought together)
def _fetch_association_rules(conn, min_support=0.01, min_confidence=0.1, top_n=15):
    # Uses the Apriori algorithm from mlxtend
    logger.info(f"Let's find some association rules! (Support > {min_support}, Confidence > {min_confidence})")
    query = '''
        SELECT 
            t.BASKET_NUM, 
            p.COMMODITY 
        FROM transactions t
        JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
        WHERE p.COMMODITY IS NOT NULL AND p.COMMODITY <> '' -- ignore missing commodity names
    '''
    try:
        df_trans = pd.read_sql(query, conn)
        if df_trans.empty:
            logger.warning("No transaction data for Apriori?")
            return []

        # Group commodities by basket
        basket_groups = df_trans.groupby('BASKET_NUM')['COMMODITY'].apply(list).values.tolist()
        
        # Apriori only cares IF an item is present, not how many. So get unique items per basket.
        basket_sets = [list(set(basket)) for basket in basket_groups]

        # Need to transform the data into a big matrix (True/False for each item in each basket)
        te = TransactionEncoder()
        te_ary = te.fit(basket_sets).transform(basket_sets)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

        # Run Apriori algorithm to find itemsets that appear frequently together
        logger.info(f"Running Apriori on {len(df_encoded)} baskets, {len(df_encoded.columns)} items...")
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            logger.warning(f"No frequent itemsets found with support > {min_support}. Maybe lower it?")
            return []

        logger.info(f"Found {len(frequent_itemsets)} frequent itemsets. Now making rules...")
        # Generate rules like "If {A} then {B}"
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if rules.empty:
            logger.warning(f"No rules found with confidence > {min_confidence}. Maybe lower it?")
            return []
            
        # Sort rules by "lift" (how much more likely B is bought if A is bought)
        rules = rules.sort_values(by='lift', ascending=False)
        
        # Convert the weird frozenset things to plain strings for the frontend
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # Pick the columns we care about and only the top N rules
        rules_output = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(top_n)

        # Round the numbers to make them look nicer
        rules_output = rules_output.round(4)
        
        logger.info(f"Generated {len(rules_output)} association rules. Nice!")
        return rules_output.to_dict(orient='records')

    except Exception as e:
        logger.error(f"Apriori blew up! Error: {e}")
        return [] # return empty list if it fails

# --- Background Task Setup --- 
precomputed_data = {} # dictionary to hold all the data for the dashboard charts
scheduler = BackgroundScheduler(daemon=True) # runs in the background

# This function runs periodically to refresh the dashboard data
async def update_dashboard_data():
    global dashboard_update_in_progress
    if dashboard_update_in_progress:
        logger.warning("Dashboard update already in progress, skipping scheduled run.")
        return
        
    logger.info("Background task starting: gonna update dashboard data...")
    dashboard_update_in_progress = True # Set flag
    # List of dashboard endpoints and the functions that get their data
    endpoints_to_update = {
        "top-spenders": _fetch_top_spenders,
        "loyalty-trends": _fetch_loyalty_trends,
        "engagement-by-income": _fetch_engagement_by_income,
        "brand-preference-split": _fetch_brand_preference_split,
        "frequent-pairs": _fetch_frequent_pairs,
        "popular-products": _fetch_popular_products,
        "seasonal-trends": _fetch_seasonal_trends,
        "churn-risk": _fetch_churn_risk,
        "association-rules": _fetch_association_rules, # the apriori results
    }
    global precomputed_data # need to modify the global dict
    conn = None
    try:
        # Get a DB connection for this update cycle
        conn = get_sqlalchemy_connection()
        if conn is None:
            logger.error("DB connection failed for background update. Skipping this run.")
            return

        temp_data = {} # store new results here temporarily
        # Loop through each endpoint and run its fetch function
        for key, fetch_func in endpoints_to_update.items():
            try:
                # Run the function to get data (e.g., _fetch_top_spenders(conn))
                temp_data[key] = fetch_func(conn)
                logger.info(f"Successfully updated data for {key}")
            except Exception as e:
                # If one fetch fails, log it but continue with others
                logger.error(f"Error updating {key} in background task: {e}")
                # Keep the old data for this key if it exists, otherwise mark error
                temp_data[key] = precomputed_data.get(key, {"error": f"Update failed: {e}"})
        # Replace the old precomputed data with the newly fetched data
        precomputed_data = temp_data
    except Exception as e:
        logger.error(f"Something went wrong during the main background update: {e}")
    finally:
        # Make sure to close the connection!
        if conn:
            conn.close()
            logger.info("Closed DB connection for background task.")
        dashboard_update_in_progress = False # Reset flag
        logger.info("Background task finished updating dashboard data.")

# Tell the scheduler to run the update function every hour
scheduler.add_job(update_dashboard_data, 'interval', hours=1, id='dashboard_update_job')
# Start the scheduler (it runs in the background)
scheduler.start()


# --- CORS Middleware --- (allows the frontend to talk to the backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # allow any origin for now (maybe restrict later?)
    allow_credentials=True,
    allow_methods=["*"], # allow all HTTP methods
    allow_headers=["*"], # allow all headers
)

# --- Password Hashing Setup ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Function to check if a plain password matches a stored hash
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Function to hash a new password
def get_password_hash(password):
    return pwd_context.hash(password)

# --- Pydantic Models --- (Define the structure of API request/response bodies)

# For CLV prediction input
class CustomerFeatures(BaseModel):
    income_range: str
    hh_size: int
    children: int

# For user registration
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

# For user login
class UserLogin(BaseModel):
    username: str
    password: str

# For basket prediction input (list of items)
class BasketItems(BaseModel):
    commodities: list[str]

# --- Helper Function --- (Just renames columns in a dataframe)
def rename_columns(df, column_map):
    # Simple rename helper
    return df.rename(columns=column_map)

# --- Helper Function --- (Turns income strings like '50-75k' into numbers)
def parse_income_range(range_str: str) -> float:
    # Tries to convert income range strings (like '50-74K') into a single number
    original_input = range_str # save for logging
    range_str = range_str.strip().lower().replace(' ', '') # clean it up
    multiplier = 1000 if 'k' in range_str else 1 # handle 'k'
    range_str = range_str.replace('k', '').replace('+', '').replace('$', '') # remove symbols

    try:
        if '-' in range_str: # like '50-74'
            low, high = map(float, range_str.split('-'))
            parsed_value = (low + high) / 2 * multiplier # use midpoint
        elif '<' in range_str: # like '<25'
            val = float(range_str.replace('<', ''))
            parsed_value = val / 2 * multiplier # guess half?
        elif '>' in range_str: # like '>150'
            val = float(range_str.replace('>', ''))
            parsed_value = val * 1.5 * multiplier # guess 1.5x?
        else: # assume it's just a number like '100'
            parsed_value = float(range_str) * multiplier
    except ValueError:
        logger.error(f"Weird income range, couldn't parse: {original_input}")
        # Fallback to 0? Or maybe raise an error?
        parsed_value = 0.0
    
    logger.info(f"Parsed income '{original_input}' into {parsed_value}")
    return parsed_value

# --- API Endpoints --- 

@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str, request: Request):
    """Handle CORS preflight OPTIONS requests manually if needed"""
    return {}

# Endpoint for registering a new user
@app.post("/register")
async def register_user(user: UserCreate, conn: pyodbc.Connection = Depends(get_pyodbc_connection)):
    # Uses the pyodbc connection cause we need direct cursor control here
    try:
        cursor = conn.cursor()
        # Check if username or email already exists
        query_check = "SELECT user_id FROM users WHERE username = ? OR email = ?"
        cursor.execute(query_check, (user.username, user.email))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username or email already taken, sorry!")

        # Hash the password before saving
        hashed_password = get_password_hash(user.password)
        # Insert the new user
        query_insert = "INSERT INTO users (username, email, hashed_password) VALUES (?, ?, ?)"
        cursor.execute(query_insert, (user.username, user.email, hashed_password))
        conn.commit() # save changes to DB
    except Exception as e:
        logger.error(f"User registration failed for {user.username}: {e}") 
        try:
            conn.rollback() # undo changes if something went wrong
        except Exception as rollback_err:
            logger.error(f"Rollback failed too!? {rollback_err}")
        raise HTTPException(status_code=500, detail="Couldn't register user, server hiccup.")
    finally:
        conn.close() # always close the connection

    return {"message": "User registered! You can log in now."}

# Endpoint for logging in
@app.post("/login")
async def login_user(user_login: UserLogin, conn: pyodbc.Connection = Depends(get_pyodbc_connection)):
    # Again, using pyodbc here
    cursor = conn.cursor()
    # Find user by username
    query = "SELECT user_id, username, hashed_password FROM users WHERE username = ?"
    cursor.execute(query, (user_login.username,))
    db_user = cursor.fetchone()
    conn.close() # close connection after query

    # If user not found or password doesn't match...
    if not db_user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    stored_hashed_password = db_user.hashed_password
    if not verify_password(user_login.password, stored_hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    # Login successful! (Normally would return a JWT token here)
    return {"message": "Login successful", "username": db_user.username}

# Endpoint for predicting Customer Lifetime Value (CLV)
@app.post("/predict-clv")
async def predict_clv(features: CustomerFeatures):
    try:
        # Turn the income string (e.g., '50-74k') into a number
        parsed_income = parse_income_range(features.income_range)

        # Need to make a DataFrame with the exact column names the model was trained on
        # Got this wrong a few times... should be income_range_numeric, HSHD_SIZE, CHILDREN
        input_df = pd.DataFrame([[parsed_income, features.hh_size, features.children]], 
                                columns=['income_range_numeric', 'HSHD_SIZE', 'CHILDREN'])
        logger.info(f"Data sent to CLV model:\n{input_df}")

        # Run the prediction (use run_in_threadpool because predict can be slow)
        prediction = await run_in_threadpool(clv_model.predict, input_df) 
        predicted_value = prediction[0] # result is usually an array

        return {
            "predicted_clv": round(predicted_value, 2) # round to 2 decimal places
        }
    
    except Exception as e:
        logger.error(f"CLV Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Couldn't predict CLV.")

# Endpoint for predicting the probability of adding the target item (e.g., DAIRY)
@app.post("/predict-target-item")
async def predict_target_item(basket: BasketItems):
    # Check if the basket prediction model actually loaded at startup
    if basket_model is None or basket_features is None:
        logger.error("Basket prediction model didn't load, can't predict!")
        raise HTTPException(status_code=503, detail="Basket prediction model isn't ready.")

    try:
        # Create a row of data matching the features the model expects, all set to 0 first
        input_data = pd.DataFrame(0, index=[0], columns=basket_features)
        
        # Now, flip the columns to 1 for the items the user actually selected
        valid_input_commodities = []
        for commodity in basket.commodities:
            if commodity in input_data.columns: # check if it's a commodity the model knows
                input_data[commodity] = 1
                valid_input_commodities.append(commodity)
            else:
                # This shouldn't happen if the frontend uses /get-prediction-features, but just in case
                logger.warning(f"User sent commodity '{commodity}' but model doesn't know it. Ignoring.")
        
        logger.info(f"Input for basket prediction (Items: {valid_input_commodities}):\n{input_data}")

        # Use predict_proba to get the probability [prob_class_0, prob_class_1]
        probabilities = await run_in_threadpool(basket_model.predict_proba, input_data)
        # We want the probability of class 1 (the target item being present)
        target_probability = probabilities[0][1] 

        logger.info(f"Predicted probability for '{TARGET_ITEM_NAME}': {target_probability:.4f}")

        return {
            "target_item": TARGET_ITEM_NAME, # tell the frontend what item we predicted
            "probability": round(target_probability * 100, 2) # return as a nice percentage
        }
    
    except Exception as e:
        logger.error(f"Basket prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Couldn't predict basket item.")

# Endpoint to give the frontend the list of commodities the basket model knows
@app.get("/get-prediction-features")
async def get_prediction_features():
    # Just return the list of feature names we loaded from the model file
    if basket_features is None:
         logger.error("Basket model features aren't loaded!")
         raise HTTPException(status_code=503, detail="Basket prediction features not available.")
    return {"features": basket_features}

# --- Data Upload Endpoints --- 

# Helper function to process uploaded CSV and append new data
async def _process_upload(file: UploadFile, table_name: str, engine):
    logger.info(f"Attempting to process uploaded file for table: {table_name}")
    contents = await file.read()
    data_io = io.BytesIO(contents)
    
    # Define primary key columns for each table (adjust if needed!)
    pk_columns = {
        'transactions': ['BASKET_NUM', 'PRODUCT_NUM', 'DATE'], 
        'households': ['HSHD_NUM'], 
        'products': ['PRODUCT_NUM'] 
    }
    
    if table_name not in pk_columns:
        logger.error(f"Primary key definition missing for table: {table_name}")
        raise HTTPException(status_code=500, detail=f"Server configuration error: PK missing for {table_name}")
        
    current_pk = pk_columns[table_name]
    
    try:
        df = pd.read_csv(data_io)
        logger.info(f"Successfully read {len(df)} rows from uploaded CSV for {table_name}.")
        
        # Ensure date column is in a consistent format if it's part of the PK
        if 'DATE' in df.columns and 'DATE' in current_pk:
            try:
                # Attempt to convert to YYYY-MM-DD format for comparison
                df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%Y-%m-%d')
            except Exception as date_err:
                 logger.warning(f"Could not standardize DATE column format: {date_err}. PK matching might be affected.")
                 # Proceed anyway, but be aware of potential mismatches

        # --- Check for existing primary keys --- 
        if not df.empty and current_pk:
            logger.info(f"Checking for existing primary keys ({', '.join(current_pk)}) in table {table_name}...")
            # Create tuples of PK values from the DataFrame for efficient lookup
            # Handle potential errors if PK columns are missing in the CSV
            try:
                pk_tuples_from_df = [tuple(x) for x in df[current_pk].to_numpy()]
            except KeyError as key_err:
                 logger.error(f"Uploaded CSV for {table_name} is missing primary key column: {key_err}")
                 raise HTTPException(status_code=400, detail=f"CSV file missing required column(s): {key_err}")

            # --- Simplified Approach: Fetch all existing keys --- 
            # Query all existing primary keys from the database table
            logger.info(f"Fetching all existing keys from DB table {table_name}...")
            all_existing_keys_query = f"SELECT DISTINCT {', '.join([f'[{col}]' for col in current_pk])} FROM {table_name}" 
            existing_keys_set = set()
            # --- Get connection inside the block --- 
            with engine.connect() as connection: 
                existing_keys_df = pd.read_sql(all_existing_keys_query, connection)
                # Convert results to a set of tuples for fast lookup
                if not existing_keys_df.empty:
                   # Ensure date column is stringified if present for proper comparison
                   if 'DATE' in existing_keys_df.columns and 'DATE' in current_pk: 
                       try:
                           existing_keys_df['DATE'] = pd.to_datetime(existing_keys_df['DATE']).dt.strftime('%Y-%m-%d')
                       except Exception as date_err_db:
                            logger.warning(f"Could not standardize DATE column from DB: {date_err_db}. PK matching might be affected.")
                   existing_keys_set = set(tuple(x) for x in existing_keys_df.to_numpy())
            logger.info(f"Found {len(existing_keys_set)} unique existing primary keys in {table_name}.")
            # --- End Simplified Approach ---

            # Create a boolean mask: True if the row's PK tuple is NOT in the existing set
            if existing_keys_set:
                mask = df.apply(lambda row: tuple(row[current_pk]) not in existing_keys_set, axis=1)
                original_rows = len(df)
                df_new = df[mask]
                rows_to_append = len(df_new)
                logger.info(f"Filtered out {original_rows - rows_to_append} rows with existing primary keys. Appending {rows_to_append} new rows.")
            else:
                 df_new = df # No existing keys found, append everything
                 logger.info("No existing keys found in DB or table is empty. Appending all rows.")
        else:
             df_new = df # Append all if df is empty or no PK defined (though PK check runs first)

        # --- Append only the new rows --- 
        if not df_new.empty:
            def append_new_data():
                # --- Get connection inside the threadpool function --- 
                with engine.connect() as connection:
                    df_new.to_sql(table_name, con=connection, if_exists='append', index=False, chunksize=1000)
                    logger.info(f"Successfully appended {len(df_new)} new rows to table: {table_name}")
            
            await run_in_threadpool(append_new_data)
        else:
             logger.info(f"No new rows to append to {table_name} after filtering.")

        # --- Trigger dashboard update --- (Moved outside the try block)
        try:
            logger.info(f"Triggering immediate run of dashboard update job after {table_name} processing.")
            scheduler.run_job('dashboard_update_job', jobstore=None)
        except Exception as job_e:
            logger.error(f"Failed to trigger immediate dashboard update job: {job_e}")
            
        return {"message": f"Processed upload for {table_name}. Appended {len(df_new) if 'df_new' in locals() else 0} new rows."}
        
    except pd.errors.EmptyDataError:
        logger.error(f"Uploaded file for {table_name} is empty.")
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file for {table_name}: {e}")
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {e}. Please ensure it's a valid CSV.")
    except HTTPException as http_exc: # Re-raise specific exceptions we handled
         raise http_exc
    except Exception as e:
        logger.error(f"Failed to process upload for table {table_name}: {e}", exc_info=True)
        # Don't close file here, finally block handles it
        raise HTTPException(status_code=500, detail=f"Server error processing file for {table_name}.")
    finally:
        # Always close the file object
        await file.close()
        logger.info(f"File object for {table_name} closed.")

@app.post("/upload/transactions")
async def upload_transactions(file: UploadFile = File(...)):
    return await _process_upload(file, "transactions", engine)

@app.post("/upload/households")
async def upload_households(file: UploadFile = File(...)):
    return await _process_upload(file, "households", engine)

@app.post("/upload/products")
async def upload_products(file: UploadFile = File(...)):
    return await _process_upload(file, "products", engine)

# --- NEW Endpoint for checking update status ---
@app.get("/dashboard-update-status")
async def get_dashboard_update_status():
    return {"updating": dashboard_update_in_progress}

# --- Dashboard Endpoints (Serving Precomputed Data) --- 
# These just return the data calculated by the background task

@app.get("/top-spenders")
async def top_spenders(): 
    # Get data from our precomputed dictionary
    data = precomputed_data.get("top-spenders")
    # Handle cases where data isn't ready yet or an error occurred during update
    if data is None:
         logger.warning("'/top-spenders' data not ready yet.")
         raise HTTPException(status_code=503, detail="Data is still calculating, try again soon!")
    if isinstance(data, dict) and "error" in data: 
         logger.error(f"Problem serving /top-spenders: {data['error']}")
         raise HTTPException(status_code=500, detail=f"Couldn't get data: {data['error']}")
    return data

@app.get("/loyalty-trends")
async def loyalty_trends(): 
    data = precomputed_data.get("loyalty-trends")
    if data is None:
         logger.warning("'/loyalty-trends' data not ready yet.")
         raise HTTPException(status_code=503, detail="Data is still calculating, try again soon!")
    if isinstance(data, dict) and "error" in data:
        logger.error(f"Problem serving /loyalty-trends: {data['error']}")
        raise HTTPException(status_code=500, detail=f"Couldn't get data: {data['error']}")
    return data

# Endpoint for the household search page (NOT precomputed)
@app.get("/household-search/{hshd_num}")
async def household_search(hshd_num: int, conn = Depends(get_sqlalchemy_connection)):
    # Fetches transactions for a specific household number when requested
    # Using SQLAlchemy's text() construct for explicit parameter handling
    query = text('''
        SELECT t.HSHD_NUM, t.BASKET_NUM, t.DATE, t.PRODUCT_NUM, p.DEPARTMENT, p.COMMODITY, t.SPEND, t.UNITS
        FROM transactions t
        JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
        WHERE t.HSHD_NUM = :hshd_param -- Use named parameter with text()
        ORDER BY t.HSHD_NUM, t.BASKET_NUM, t.DATE, t.PRODUCT_NUM, p.DEPARTMENT, p.COMMODITY;
    ''')
    try:
        # Pass parameters as a dictionary with text()
        df = pd.read_sql(query, conn, params={'hshd_param': hshd_num})
        # Rename columns to match what the frontend expects
        df = rename_columns(df, {
            'HSHD_NUM': 'Hshd_num',
            'BASKET_NUM': 'Basket_num',
            'DATE': 'Date',
            'PRODUCT_NUM': 'Product_num',
            'DEPARTMENT': 'Department',
            'COMMODITY': 'Commodity',
            'SPEND': 'Spend',
            'UNITS': 'Units'
        })
    except Exception as e:
        logger.error(f"Error searching household {hshd_num}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching data")
    finally:
        # Close the connection (might be handled automatically, but good practice?)
        if conn:
             conn.close() 
    if df.empty:
         # Return an empty list if the household has no transactions
         logger.info(f"No transactions found for HSHD_NUM: {hshd_num}. Returning empty list.")
         return []
    # Convert the resulting DataFrame to a list of dictionaries for JSON
    return df.to_dict(orient='records')

@app.get("/engagement-by-income")
async def engagement_by_income(): 
    data = precomputed_data.get("engagement-by-income")
    if data is None:
         logger.warning("'/engagement-by-income' data not ready yet.")
         raise HTTPException(status_code=503, detail="Data is still calculating, try again soon!")
    if isinstance(data, dict) and "error" in data:
        logger.error(f"Problem serving /engagement-by-income: {data['error']}")
        raise HTTPException(status_code=500, detail=f"Couldn't get data: {data['error']}")
    return data

@app.get("/brand-preference-split")
async def brand_preference_split(): 
    data = precomputed_data.get("brand-preference-split")
    if data is None:
         logger.warning("'/brand-preference-split' data not ready yet.")
         raise HTTPException(status_code=503, detail="Data is still calculating, try again soon!")
    if isinstance(data, dict) and "error" in data:
        logger.error(f"Problem serving /brand-preference-split: {data['error']}")
        raise HTTPException(status_code=500, detail=f"Couldn't get data: {data['error']}")
    return data

@app.get("/frequent-pairs")
async def frequent_pairs(): 
    data = precomputed_data.get("frequent-pairs")
    if data is None:
         logger.warning("'/frequent-pairs' data not ready yet.")
         raise HTTPException(status_code=503, detail="Data is still calculating, try again soon!")
    if isinstance(data, dict) and "error" in data:
        logger.error(f"Problem serving /frequent-pairs: {data['error']}")
        raise HTTPException(status_code=500, detail=f"Couldn't get data: {data['error']}")
    return data

@app.get("/popular-products")
async def popular_products(): 
    data = precomputed_data.get("popular-products")
    if data is None:
         logger.warning("'/popular-products' data not ready yet.")
         raise HTTPException(status_code=503, detail="Data is still calculating, try again soon!")
    if isinstance(data, dict) and "error" in data:
        logger.error(f"Problem serving /popular-products: {data['error']}")
        raise HTTPException(status_code=500, detail=f"Couldn't get data: {data['error']}")
    return data

@app.get("/seasonal-trends")
async def seasonal_trends(): 
    data = precomputed_data.get("seasonal-trends")
    if data is None:
         logger.warning("'/seasonal-trends' data not ready yet.")
         raise HTTPException(status_code=503, detail="Data is still calculating, try again soon!")
    if isinstance(data, dict) and "error" in data:
        logger.error(f"Problem serving /seasonal-trends: {data['error']}")
        raise HTTPException(status_code=500, detail=f"Couldn't get data: {data['error']}")
    return data

@app.get("/churn-risk")
async def churn_risk(): 
    # Returns the list of at-risk customers and summary stats
    data = precomputed_data.get("churn-risk")
    if data is None or not isinstance(data, dict) or "at_risk_list" not in data:
        logger.warning("'/churn-risk' data not ready or wrong format.")
        raise HTTPException(status_code=503, detail="Data is still calculating, try again soon!")
    if isinstance(data, dict) and "error" in data:
        logger.error(f"Problem serving /churn-risk: {data['error']}")
        raise HTTPException(status_code=500, detail=f"Couldn't get data: {data['error']}")
    return data

@app.get("/association-rules")
async def association_rules_endpoint(): 
    # Returns the Apriori rules
    data = precomputed_data.get("association-rules")
    if data is None: 
        logger.warning("'/association-rules' data not ready yet.")
        raise HTTPException(status_code=503, detail="Data is still calculating, try again soon!")
    if isinstance(data, dict) and "error" in data: 
        logger.error(f"Problem serving /association-rules: {data['error']}")
        raise HTTPException(status_code=500, detail=f"Couldn't get data: {data['error']}")
    # Empty list is okay, frontend should handle it
    return data

# --- Run the App! --- 
if __name__ == "__main__":
    import uvicorn
    # Use reload=True for development, makes life easier
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
