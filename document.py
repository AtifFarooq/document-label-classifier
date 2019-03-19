# Manage necessary imports
import pandas as pd
import numpy as np
import csv
import networkx as nx
import matplotlib.pyplot as plt
import collections
from node2vec import Node2Vec
from nltk import word_tokenize

%matplotlib inline

# construct a networkx graph
G = nx.Graph()

# open a local .txt file & add edges
with open('links.txt', 'r') as file:
    for line in file:
        link = line.split()
        if link[0] not in G:
            G.add_node(link[0])
        if link[1] not in G:
            G.add_node(link[1])
        G.add_edge(link[0], link[1])

# generate walks
node2vec = Node2Vec(G, dimensions=200, walk_length=15, num_walks=40)

# Learn the embeddings
model = node2vec.fit(window=10, min_count=1)
print(type(model))

from gensim.models import Word2Vec
from sklearn import preprocessing

content_file_path = 'content.txt'

def extract_data(content_file_path):
    '''
        Extract all the data provided in the file content.txt
    '''
    document_ids = []
    data = []
    node_to_words = dict()

    with open(content_file_path, 'r') as content_file:
        for line in content_file.readlines():
            tokens = line.split()
            document_id = tokens[0]
            text = tokens[1:]
            document_ids.append(document_id)
            data.append(text)
            node_to_words[document_id] = text
        print('number of nodes is %s. ' % len(document_ids))
        return (document_ids, data, node_to_words)

def get_mapping(document_ids):
    '''
        Mapping that uses a label encoder. Not used in this implementation.
    '''
    le = preprocessing.LabelEncoder()
    le.fit(document_ids)
    idx = le.transform(document_ids)
    document_ids = le.inverse_transform(idx)
    return idx, document_ids

def get_node_to_id(document_ids):
    '''
        return a mapping from each node/document_id to an id
        number, starting with index 0
    '''
    node_to_id = dict()
    for node in document_ids:
        node_to_id[node] = len(node_to_id)
    return node_to_id

def vectorize_words(textual_data):
    '''
       train word2vec on the textual data
    '''
    word_to_vector = Word2Vec(textual_data, size=200, window=5, min_count=1)
    word_to_vector.train(textual_data, total_examples=len(textual_data), epochs=20)
    return word_to_vector

# extract all relevant data from the given file
document_ids, textual_data, node_to_words = extract_data(content_file_path)
node_to_id = get_node_to_id(document_ids)

# print list of words corresponding to node/document id 31336
print("The words in document_id 31336:", node_to_words['31336'])

# print vector corresponding to 'w125' for testing
word_to_vector = vectorize_words(textual_data)
# print("The vector representation of w125", word_to_vector['w125'])

def sentence_matrix(list_of_words):
    '''
        Given a list of words, lookup the corresponding
        word vectors for each word, and return an ndmatrix
        with this information
    '''
    list_of_vectors = [word_to_vector[word] for word in list_of_words]
    sentence_matrix = np.stack(list_of_vectors, axis=0)
    return sentence_matrix

def get_mean_vector(sentence_matrix):
    '''
        Takes in a sentence matrix, averages across axis=0 and returns
    '''
    mean_vector = np.mean(sentence_matrix, axis=0)
    return mean_vector

def generate_node_to_vector(node_to_words):
    '''
        transform the node_to_words (dict) mapping into a node -> dense vector mapping,
        (where the vector only contains textual information)
    '''
    node_to_vector = dict()
    for node in node_to_words.keys():
        list_of_words = node_to_words[node]
        matrix = sentence_matrix(list_of_words)
        mean_vector = get_mean_vector(matrix)
        node_to_vector[node] = mean_vector
    return node_to_vector

node_to_vector = generate_node_to_vector(node_to_words)
# print(node_to_vector['31336'])
# print("Shape of node_to_vector: ", len(node_to_vector))

def concatenate_arrays(a, b):
    return np.concatenate([a,b])

def generate_node_embeddings(node_to_vector):
    '''
        transform the node_to_vector dictionary into a node -> dense vector mapping,
        (where the vector contains both the network embedding + textual information)
    '''
    node_embeddings = dict()
    for node in node_to_vector.keys():
        a = node_to_vector[node]
        b = model[node]
        embedding = concatenate_arrays(a, b)
        node_embeddings[node] = embedding
    return node_embeddings

node_embeddings = generate_node_embeddings(node_to_vector)
print("Concatenated node to vector embedding corresponding to node/doc id 31336:")
# print(node_embeddings['31336'])
print(node_embeddings['31336'].shape)

# Here, we generate an embeddings matrix corresponding to all unique nodes/documents in our network

num_of_nodes = len(node_embeddings)
embedding_dimension = max([len(embedding) for embedding in node_embeddings.values()])

def generate_embeddings_matrix(num_of_nodes, embedding_dimension):
    embeddings_matrix = np.zeros((num_of_nodes, embedding_dimension))
    for node in node_embeddings.keys():
        index = node_to_id[node]
        embedding_dense_vector = node_embeddings[node]
        embeddings_matrix[index] = embedding_dense_vector
    return embeddings_matrix

embeddings_matrix = generate_embeddings_matrix(num_of_nodes, embedding_dimension)
print("Generated Embeddings Matrix")
print("Shape of the Embeddings Matrix is: ", embeddings_matrix.shape)
# print("The dense vector corresponding to the entry at index 0 is: ", embeddings_matrix[0])

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional

def read_training_dataset(file_name):
    with open(file_name, 'r') as train_data_file:
        reader = csv.reader(train_data_file)
        # ignore the first row
        next(reader)
        data = []
        data_labels = []

        for row in reader:
            data.append(row[0])
            data_labels.append(row[1])
        X_train = np.asarray(data)
        data_labels = [(int(label)-1) for label in data_labels]
        y_train = to_categorical(data_labels, num_classes=7)
        return (X_train, y_train)

def read_test_dataset(file_name):
    with open(file_name, 'r') as test_data_file:
        reader = csv.reader(test_data_file)
        next(reader)
        data = []

        for row in reader:
            data.append(row[0])
        X_train = np.asarray(data)
        return X_train

X_train, y_train = read_training_dataset('label.train.csv')
print("Shape of X_train: ", X_train.shape)
print("Shape of y_train: ", y_train.shape)
print(type(X_train))
print(type(y_train))

new_x_train = []
for node in X_train:
    new_id = node_to_id[node]
    new_x_train.append(new_id)
X_train = new_x_train

X_test = read_test_dataset('label.test.csv')
print("Shape of X_test: ", X_test.shape)
print(type(X_test))

# Define the embedding layer. This will be the first layer of the network
# We set 'trainable' to False as we don't want these pre-trained weights to change
embedding = Embedding(num_of_nodes, embedding_dimension, weights=[embeddings_matrix], input_length=1, trainable=False)

# Define the learning model
model = Sequential()
model.add(embedding)
model.add(Dropout(0.2))
model.add(LSTM(embedding_dimension))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))
model.compile(loss ='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())

# train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Test the model on the given test set and determine the predicted classes
id_numbers = X_test

new_x_test = []
for node in X_test:
    new_id = node_to_id[node]
    new_x_test.append(new_id)
X_test = new_x_test

y_predicted = model.predict_classes(X_test)
print(y_predicted)

y_predicted = [(int(label)+1) for label in y_predicted]
print(y_predicted)

# Write results about predictions to .csv file
with open('predicted_classes.csv', 'w') as file:
    file.write('id' + ',' + 'label' + '\n')
    for i in range(len(y_predicted)):
        row = str(id_numbers[i]) + "," + str(y_predicted[i])
        file.write(row)
        file.write('\n')
