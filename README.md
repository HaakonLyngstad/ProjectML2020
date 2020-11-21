# ProjectML2020


 
**This project is divided into two parts**

1. The first is a group of machine learning algorithtms and classifiers, with datasets available for training a NLP prediction model.
2. The second is a django project, a website with a demo of how an open source project such as this could be used.

## How to set up the project

**NB! This is a guide for setting up on Windows. Other OS's (Linux and Mac) will have some other commands**

This project requires that you have Python 3 installed on your device, and added as an environment variable

1. Clone this repository to a location of your choosing on your device. 
  - This can be done in the coommand shell by writing *git clone https://github.com/HaakonLyngstad/ProjectML2020.git*
  
2. cd into the project
3. A virtual environment is reccomended, it usually installs with python, if not write *pip install virtualenv*
3. Write *virtualenv venv* to create a virtual environment
4. Write *venv\Scripts\activate*. You should now be inside a virtual enviroment (look for the (vevn) mark to the left of you command line
5. Install all required libraries with *pip install -r "requirements.txt"* - this can take a while

You should now be able to train your own models by running the "main.py" script (the one in the root of the repositry).

### Setting up the Django website

1. Make sure you are in the root of the repository
2. cd into "mlwebsite"
3. Write *python manage.py makemigrations* followed by *python manage.py migrate*.
4. Write *python manage.py runserver* and go to http://127.0.0.1:8000/ in a browser
5. You should now be able to access the website

## Code style

[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg?style=flat)](https://github.com/feross/standard)


## License
This project is publicy available under the MIT license

MIT © [Time_series_group_3]()
