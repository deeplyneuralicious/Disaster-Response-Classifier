# Disaster Response Pipeline Project

## Project Description

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster_Response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Disaster_Response.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app in localhost: `python run.py`

4. Copy the localhost link on browser to view the web app.
