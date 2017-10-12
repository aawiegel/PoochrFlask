# Copyright 2015 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from poochr import storage

from flask import Flask, current_app, redirect, render_template, request, \
    url_for

from google.cloud import error_reporting
import google.cloud.logging
import httplib2
from oauth2client.contrib.flask_util import UserOAuth2

import os

import numpy as np

from werkzeug.utils import secure_filename

import pickle

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.xception import preprocess_input
from keras import backend as K
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

model = load_model('xbreeds_model.h5')
breed_to_index = np.load('breed_indices.npy').tolist()
index_to_breed = {v: k for k, v in breed_to_index.items()}
max_breed_mat = np.load('breed_max_matrix.npy')
with open('dog_vocabulary.p', 'rb') as file:
    dog_vocab = pickle.load(file)
with open('dog_lsa_matrix.npy', 'rb') as file:
    dog_lsa_mat = np.load(file)
with open('dogtime_urls.p', 'rb') as file:
    dogtime_url_dict = pickle.load(file)

generate_dog_features = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[-2].output])

dog_tfidf = TfidfVectorizer(vocabulary=dog_vocab, ngram_range = (1,2))

with open('breed_lsa.p', 'rb') as file:
    breed_lsa = pickle.load(file)

def get_model():
    model_backend = current_app.config['DATA_BACKEND']
    if model_backend == 'cloud    @app.errorhandler(500)
    def server_error(e):
        client = error_reporting.Client(app.config['PROJECT_ID'])
        client.report_exception(
            http_context=error_reporting.build_flask_context(request))
        return """
        An internal error occurred.
        """, 500sql':
        from . import model_cloudsql
        model = model_cloudsql
    elif model_backend == 'datastore':
        from . import model_datastore
        model = model_datastore
    elif model_backend == 'mongodb':
        from . import model_mongodb
        model = model_mongodb
    else:
        raise ValueError(
            "No appropriate databackend configured. "
            "Please specify datastore, cloudsql, or mongodb")

    return model

def upload_image_file(file):
    """
    Upload the user-uploaded file to Google Cloud Storage and retrieve its
    publicly-accessible URL.
    """
    if not file:
        return None

    public_url = storage.upload_file(
        file.read(),
        file.filename,
        file.content_type
    )

    current_app.logger.info(
        "Uploaded file %s as %s.", file.filename, public_url)

    return public_url

#def allowed_file(filename):
#    return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def create_app(config, debug=False, testing=False, config_overrides=None):
    app = Flask(__name__)
    app.config.from_object(config)

    app.debug = debug
    app.testing = testing

    if config_overrides:
        app.config.update(config_overrides)

    if not app.testing:
        client = google.cloud.logging.Client(app.config['PROJECT_ID'])
        # Attaches a Google Stackdriver logging handler to the root logger
        client.setup_logging(logging.INFO)

    # Setup the data model.
    with app.app_context():
        model = get_model()
        model.init_app(app)

    # Register the Bookshelf CRUD blueprint.
    # from .crud import crud
    # app.register_blueprint(crud, url_prefix='/')

    # Add a default root route.
    @app.route("/")
    def index():
        return render_template('index.html')

    @app.route('/predict', methods=['GET', 'POST'])
    def predict_file():
        if request.method == 'POST':
            data = request.form.to_dict(flat=True)

            if len(data['desc']) > 1000:
                flash('Shorten your text description to less than 1000 characters.')
                return redirect(url_for('index'))
            if 'image' not in request.files:
                flash('No file part')
                return redirect(url_for('index'))
            # If an image was uploaded, update the data to point to the new image.
            # [START image_url]
            image_url = upload_image_file(request.files.get('image'))
            # [END image_url]
            # [START image_url2]
            if image_url:
                data['imageUrl'] = image_url

            # [END image_url2]

            messages = [data['desc']]

            img = image.load_img(request.files.get('image'),
                                 target_size=(299, 299))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            dog_vec = generate_dog_features([img, 0])

            best_guess = np.argmax(dog_vec)
            if best_guess == 114:
                messages.append("""Are you sure this is a dog?
                                Well, I'll give you recommendations, but I
                                doubt they'll be good.""")



            data['best_guess'] = int(best_guess)
            data['best_guess_str'] = index_to_breed[best_guess]

            dog_text_vec = dog_tfidf.fit_transform([data['desc']])
            dog_lsa_vec = breed_lsa.transform(dog_text_vec)

            image_similarity = cosine_similarity(max_breed_mat, dog_vec[0])
            text_similarity = cosine_similarity(dog_lsa_mat, dog_lsa_vec)

            weight = 0.5
            combined_sim = weight*image_similarity + (1-weight)*text_similarity
            guesses = np.argsort(combined_sim.T)[0][::-1]

            labels = [index_to_breed[guess].split("-", 1)[0] for guess in guesses[:3]]
            breeds = [index_to_breed[guess].split("-", 1)[1] for guess in guesses[:3]]
            urls = [dogtime_url_dict[breed.lower()] for breed in breeds]
            breeds = [url.split("/")[-1].replace("-", " ").replace("_", " ").title() \
                      for url in urls]

            data['recommendations'] = [int(guess) for guess in guesses[:3]]
            data['recommendations_str'] = breeds

            labels_breeds_urls = zip(labels, breeds, urls)
            for item in data.values():
                print(type(item))
            dog = get_model().create(data)

            return render_template('predict_image.html',
                                   image_url=image_url, messages=messages,
                                   labels_breeds_urls = labels_breeds_urls)
        return redirect(url_for('index'))


    # Add an error handler. This is useful for debugging the live application,
    # however, you should disable the output of the exception for production
    # applications.
    @app.errorhandler(500)
    def server_error(e):
        client = error_reporting.Client(app.config['PROJECT_ID'])
        client.report_exception(
            http_context=error_reporting.build_flask_context(request))
        return """
        An internal error occurred.
        """, 500

    return app
