# Disaster Response Pipeline Project

## Project Description
In the midst of a disaster events, time is of essence when it comes to providing swift response and medical attention to the victims. With the boooming of social media, poeple tend to leverage social media to seek rescue by sending SOS messages to the platform, however identifying one relevant message is often challenging like looking a needle in a haystack. Thus, a classifier can be implemented in this application to identify if the messages is sent from the victim and then the targeted messages can be directed to an appropriate disaster relief agency.

In this project, a disaster pipeline classifer is created to classify the disaster related messages into 36 different categories in order to pinpoint what kind of aid is needed for the respective senders. Here, natural language processing technique will be used to tokenize the text messages and feature-engineered in order to be trained in a classifer on top of the categorical label data. One thing to note is each message can be assigned into multiple categories based on the nature of the message. ETL and ML pipelines are created as well to perform data preprocessing on the raw data and model training and tuning separately. Lastly, the trained model will be serialised as pickle file and then deployed to web app using Flask for live prediction.

![](https://github.com/deeplyneuralicious/Disaster-Response-Classifier/blob/main/Screenshots/WebAppChart.png)

![](https://github.com/deeplyneuralicious/Disaster-Response-Classifier/blob/main/Screenshots/WebAppMessageClassifier.png)

### Instructions for Running the Files:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster_Response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Disaster_Response.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app in localhost: `python run.py`

4. Copy the localhost link on browser to view the web app.

### Acknowledgement
- Udacity
- Future Eight
