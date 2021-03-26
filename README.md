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

# Knowledge Graph Usage
First, install requirements:

```bash
pip install requirements.txt
```

Set the environment variables `FLASK_APP` and `FLASK_ENV`: 

```bash
export FLASK_APP=endpoint
export FLASK_ENV=development
```

or, if on Windows:

```bash
set FLASK_APP=endpoint
set FLASK_ENV=development
```

Spin up the Flask App

```bash
flask run
```

Request bodies for this need to be computed within Python with spaCy, so run:

```bash
python requestlauncher.py [filename]
```

where the file is a list of URLs.

The app will print out (in the console) a similarity matrix where entry (i,j) is the number of common domain terms between URL i and URL j.
