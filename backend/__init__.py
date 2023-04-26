from flask import Flask
import os
#from config import Config
app = Flask(__name__,static_folder = os.path.join(os.getcwd(), 'data'))
#app.config.from_object(Config)
from backend import routes