# semantic-web-mining
CSE 573 Semantic Web Mining: Browser Extension

Tutorial:
https://developer.chrome.com/docs/extensions/mv2/getstarted/

Demo:
https://drive.google.com/file/d/12ZdymfL8-B7MYO3II4qDbg4afHWjnWSF/view?usp=sharing

Tasks:
- Basics: 
    - get started with chrome tutorial, have working app to build off of
    - app should be able to exit developer mode/distributable  
- Acquire data:
    - read user search queries
    - store in internal data struct (maybe a json which has search query + first 15 result)
    - track which search results were clicked on
- Ratings:
    - provide option for like/dislike one of the search results (beside the hyperlink which google displays?)
- Credentials:
    - user login page+interface
    - user login backend w/ access to data
- Database:
    - acquired data to DB

-------------------------------------------------------

## Run Flask App: 
1. Navigate to folder containing echosrv.py (flask app in /flaskplay/) 
    a. If you do not have flask installed run: 
        pip install flask 
2. RUN: 
    export FLASK_APP=echosrv.py
3. RUN: 
    flask run --host=0.0.0.0



