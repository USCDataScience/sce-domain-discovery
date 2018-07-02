from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
import flask
import numpy as np
import os
import json
from flask import request, flash
from sklearn.neural_network import MLPClassifier

accuracy = 0.0
splits = 2
iteration = 1

#def read_config(file, section):
#    Config = configparser.ConfigParser()
#    Config.read(file)
#    configs = {}
#    options = Config.options(section)
#    for option in options:
#        configs[option] = Config.get(section, option)
#    return configs

def load_vocab():
    """Load Vocabulary"""
    #with open("/Users/ksingh/git-workspace/dd-polar/seedexplorer/src/main/resources/data/keywords.txt", 'rb') as f:
    if os.path.exists('keywords.txt'):
        with open("keywords.txt", 'rb') as f:
            keywords_content = f.read()
    else:
        with open("keywords.txt", 'wb') as fw:
            fw.write("This is a test")
            keywords_content = "This is a test"
    count_vect = CountVectorizer(lowercase=True, stop_words='english')
    count_vect.fit_transform([keywords_content])
    keywords = count_vect.vocabulary_
    return keywords

def clear_model():
    print('clear_model')
    fname = 'model.pkl'
    if os.path.isfile(fname):
        os.remove(fname)
    setattr(flask.current_app, 'model', None)
    return '0'

def select_model(mName):
    #app/classifier/
    file = open('model.conf', 'w')
    file.write("[Model]\n")
    file.write("model="+mName)
    file.close()
    return ' Model is Selected!'

def update_model(annotations):
    global accuracy, splits, iteration
    modelName = 'Naive Bayes'
#    if os.path.exists('model.conf'):
#        name = read_config('model.conf', 'Model')
#        modelName = name.get('model')
    if modelName == 'NN':
        url_text = getattr(flask.current_app, 'url_text', None)
        clf = MLPClassifier(max_iter=1000, learning_rate='adaptive',)
        count_vect = CountVectorizer(lowercase=True, stop_words='english')
        tfidftransformer = TfidfTransformer()

        if url_text is None:
            print('An error occurred while accessing the application context variables')
            return '-1'

        labeled = np.array(annotations)
        model=getattr(flask.current_app, 'model', None)

        if model is not None:
            # add the old docs to the new
            prev_url_text=model['url_text']
            prev_labeled=model['labeled']
            url_text=np.append(url_text,prev_url_text,axis=0)
            labeled=np.append(labeled,prev_labeled,axis=0)

        features = count_vect.fit_transform(url_text)
        features=tfidftransformer.fit_transform(features).toarray().astype(np.float64)
        print('No. of features: ' + str(len(features)) + ' and No. of labels: ' + str(len(labeled)))
        print np.unique(labeled)
        clf.fit(features, labeled,)

        # save the model
        model={'url_text':url_text,'labeled':labeled,'countvectorizer':count_vect,'tfidftransformer':tfidftransformer,'clf':clf}
        setattr(flask.current_app, 'model', model)
        predicted = clf.predict(features)
        accuracy = (labeled == predicted).sum() / float(len(labeled))
        fname = 'model.pkl'
        joblib.dump(model, fname)
    
    
    if modelName == 'Naive Bayes':
        clf = GaussianNB()
        model=getattr(flask.current_app, 'model', None)
        keywords={}
        if model is not None:
            # add the old docs to the new
            prev_url_text=model['url_text']
            prev_labeled=model['labeled']
            url_text=np.append(url_text,prev_url_text,axis=0)
            labeled=np.append(labeled,prev_labeled,axis=0)
            keywords =model['keywords']
            if keywords is None:
                keywords = load_vocab()
        
        count_vect = CountVectorizer(lowercase=True, stop_words='english', vocabulary=keywords.keys())
        url_text = getattr(flask.current_app, 'url_text', None)
        if url_text is None or clf is None:
            print('An error occurred while accessing the application context variables')
            return '-1'
        
        features = count_vect.fit_transform(url_text).toarray().astype(np.float64)
        labeled = np.array(annotations)
        print('No. of features: ' + str(len(features)) + ' and No. of labels: ' + str(len(labeled)))
        clf.partial_fit(features, labeled, classes=np.unique(labeled))
        predicted = clf.predict(features)
        accuracy = (labeled == predicted).sum() / float(len(labeled))
         # save the model
        model={'url_text':url_text,'labeled':labeled,'countvectorizer':count_vect,'clf':clf}
        setattr(flask.current_app, 'model', model)
       
        
    if modelName == 'SVM':
        clf = linear_model.SGDClassifier()
        model=getattr(flask.current_app, 'model', None)
        if model is not None:
            # add the old docs to the new
            prev_url_text=model['url_text']
            prev_labeled=model['labeled']
            url_text=np.append(url_text,prev_url_text,axis=0)
            labeled=np.append(labeled,prev_labeled,axis=0)
            keywords =model['keywords']

        if keywords is None:
            keywords = load_vocab()
        
        count_vect = CountVectorizer(lowercase=True, stop_words='english', vocabulary=keywords.keys())
        url_text = getattr(flask.current_app, 'url_text', None)
        if url_text is None or clf is None:
            print('An error occurred while accessing the application context variables')
            return '-1'
        
        features = count_vect.fit_transform(url_text).toarray().astype(np.float64)
        labeled = np.array(annotations)
        print('No. of features: ' + str(len(features)) + ' and No. of labels: ' + str(len(labeled)))
        clf.partial_fit(features, labeled, classes=np.unique(labeled))
        predicted = clf.predict(features)
        accuracy = (labeled == predicted).sum() / float(len(labeled))
         # save the model
        model={'url_text':url_text,'labeled':labeled,'countvectorizer':count_vect,'clf':clf}
        setattr(flask.current_app, 'model', model)
    

    dictionary = get_metrics(model)
    json_dictionary = json.dumps(dictionary)

    return json_dictionary



def get_metrics(model):
    unique, counts = np.unique(model['labeled'], return_counts=True)
    dictionary = dict(zip(unique, counts))

    return dictionary


def predict(txt):


    model = getattr(flask.current_app, 'model', None)

    if model is None:
        return -1

    count_vect = model['countvectorizer']
    tfidftransformer = model['tfidftransformer']
    clf=model['clf']

    features = count_vect.transform([txt])
    features=tfidftransformer.transform(features).toarray().astype(np.float64)

    predicted = clf.predict(features)
    print(predicted)
    return predicted[0]

def import_model():
    global accuracy
    print 'importing'
    filename = 'model.pkl'


    if 'file' not in request.files:
        flash('No file part')
        return '-1'
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return '-1'
    if file:
        # filename = secure_filename(file.filename)
        file.save(os.path.join(flask.current_app.root_path, flask.current_app.config['UPLOAD_FOLDER'], filename))
        # return redirect(url_for('uploaded_file', filename=filename))
    else:
        flash('An error occurred while uploading the file')
        return '-1'

    model = joblib.load(os.path.join(flask.current_app.root_path, flask.current_app.config['UPLOAD_FOLDER'], filename))

    accuracy = model['accuracy']

    setattr(flask.current_app, 'model', model)

    dictionary = get_metrics(model)
    json_dictionary = json.dumps(dictionary)

    # return str(accuracy)
    return json_dictionary


def export_model():
    global accuracy
    model = getattr(flask.current_app, 'model', None)

    if model is None:
        return -1

    model['accuracy']=accuracy

    fname = 'model.pkl'
    joblib.dump(model, fname)

    return flask.send_from_directory(directory=flask.current_app.root_path + '/../', filename=fname)


def check_model():
    model = getattr(flask.current_app, 'model', None)
    if model is None:
        return str(-1)

    dictionary = get_metrics(model)
    json_dictionary = json.dumps(dictionary)

    # return str(0)
    return json_dictionary
