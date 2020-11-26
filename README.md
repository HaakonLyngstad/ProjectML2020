# ProjectML2020


 
**This project is divided into two parts**

1. The first is a group of machine learning algorithtms and classifiers, with datasets available for training a NLP prediction model.
2. The second is a django project, a website with a demo of how an open source project such as this could be used.

## How to set up the project

**NB! This is a guide for setting up on Windows. Other OS's (Linux and Mac) will have some other commands**

This project requires that you have Python 3.6.8 installed on your device, and added as an environment variable. 
Note: Other python version might be sufficient, but this has not been tested.

1. Clone this repository to a location of your choosing on your device. 
  - This can be done in the coommand shell by writing *git clone https://github.com/HaakonLyngstad/ProjectML2020.git*
  
2. cd into the project
3. A virtual environment is reccomended, it usually installs with python, if not write *pip install virtualenv*
3. Write *virtualenv venv* to create a virtual environment
4. Write *venv\Scripts\activate*. You should now be inside a virtual enviroment (look for the (vevn) mark to the left of you command line
5. Install all required libraries with *pip install -r "requirements.txt"* - this can take a while
6. cd into "mlwebsite"

You should now be able to train your own models by running the "main.py" script (the one in the root of the repositry).

### Setting up the Django website

The website is an example of how a online fraud detection text classifier could be implemented in a real use case. It also explaines our methods in manner
suited for individuals without prior knowledge and/or education on how the used machine learning models functions. 

1. Make sure you are in the "mlwebsite" folder
3. Write *python manage.py makemigrations* followed by *python manage.py migrate*.
4. Write *python manage.py runserver* and go to http://127.0.0.1:8000/ in a browser
5. You should now be able to access the website

### Website presentation video

If for some reason an interested party should find themselves unable to properly clone, install and run the website
on their local machines, this repository also contains a short video showcasing the intended usage of the website. 

## Code style

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![ForTheBadge uses-html](http://ForTheBadge.com/images/badges/uses-html.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-css](http://ForTheBadge.com/images/badges/uses-css.svg)](http://ForTheBadge.com)
[![ForTheBadge uses-js](http://ForTheBadge.com/images/badges/uses-js.svg)](http://ForTheBadge.com)


## License
This project is publicy available under the MIT license

MIT © [Time_series_group_3]()
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

