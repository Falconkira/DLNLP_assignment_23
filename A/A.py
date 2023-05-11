import matplotlib.pyplot as plt
import os
import re
import tensorflow as tf                                                                             
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,log_loss
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import layers
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import re
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

import string
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def data_preprocessing():
    """
    This function read the dataset and split to train, validation and test set.
    Retrun:
        train, val, test: Dataframe of the dataset
    """
    print("Data Preprocessing......")
    data = pd.read_csv('Datasets/train.csv')
    data["essay_text"] = data["essay_id"].apply(lambda x: open(f'Datasets/train/{x}.txt').read())
    data["discourse_effectiveness"] = data["discourse_effectiveness"].map({
        "Adequate":1,
        "Effective":2,
        "Ineffective":0
    })

    #data = pd.DataFrame(data,columns = ['discourse_text', 'discourse_effectiveness'])
    train,temp = train_test_split(data, test_size = 0.2, stratify = data['discourse_effectiveness'])
    val,test = train_test_split(temp, test_size = 0.5)

    return train, val, test

def plot_graphs(history, metric):
    """
    This function plot the metric vs epochs graph for the model.
    Args:
        history: the TensorFlow training history
        metric: a string of the metric want to plot(accuracy, loss)
    Retrun:
        train, val, test: Dataframe of the dataset
    """
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


def SVM(train, test):
    """
    This function preprocss and train the SVM model for classfication.
    Args:
        train: the training set in dataframe
        test: the testing set in dataframe
    Retrun:
        acc: a float number of training accuracy of the model
        acc_test: a float number of testing accuracy of the model
    """
    print("Runing the SVM model......")
    #implementation of SVM model
    #acc:0.6410116943160185 precision:0.6383289331860126 recall:0.6410116943160185 loss:0.7991590490738599
    #TF-IDF to vectorize the text
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train["discourse_text"])
    xtrain = vectorizer.transform(train["discourse_text"])
    xtest = vectorizer.transform(test["discourse_text"])
    ytrain = train["discourse_effectiveness"]
    ytest = test["discourse_effectiveness"]

    #SVM
    classifier = svm.SVC(kernel = "linear", probability = True).fit(xtrain, ytrain)
    svm_pred = classifier.predict(xtrain)
    #print("accuracy", accuracy_score(ytrain, svm_pred))
    acc = accuracy_score(ytrain, svm_pred)
    svm_pred = classifier.predict(xtest)
    #print("accuracy", accuracy_score(ytest, svm_pred))
    acc_test = accuracy_score(ytest, svm_pred)
    #print("precision", precision_score(ytest, svm_pred, average = "weighted"))
    #print("recall", recall_score(ytest, svm_pred, average = "weighted"))
    pred = classifier.predict_proba(xtest)
    #print("log", log_loss(ytest, pred))

    return acc, acc_test

def createCNN(train, val ,test):
    """
    This function preprocss the data and create the CNN model for classfication.
    Args:
        train: the training set in dataframe
        val : the validation set in dataframe
        test: the testing set in dataframe
    Retrun:
        model: complied model ready for training
        train_dataset, val_dataset, test_dataset: preprocssed dataset for training and testing
    """
    #CNN
    #acc: 0.6576013054120207 precision: 0.6595459988687097 recall: 0.6576013054120207 loss: 0.7871657
    # Tokenize the text and convert to sequences
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(train["discourse_text"])
    train_sequences = tokenizer.texts_to_sequences(train["discourse_text"])
    val_sequences = tokenizer.texts_to_sequences(val["discourse_text"])
    test_sequences = tokenizer.texts_to_sequences(test["discourse_text"])

    # Pad the sequences
    max_len = 200
    train_data = pad_sequences(train_sequences, maxlen=max_len)
    val_data = pad_sequences(val_sequences, maxlen=max_len)
    test_data = pad_sequences(test_sequences, maxlen=max_len)

    # Convert the labels to one-hot encoded format
    train_labels = tf.keras.utils.to_categorical(train["discourse_effectiveness"], num_classes=3)
    val_labels = tf.keras.utils.to_categorical(val["discourse_effectiveness"], num_classes=3)
    test_labels = tf.keras.utils.to_categorical(test["discourse_effectiveness"], num_classes=3)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(len(train_data)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)

    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=max_len),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=3, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model, train_dataset, val_dataset, test_dataset


def trainCNN(model, train_dataset, val_dataset):
    """
    This function trains the created CNN model.
    Args:
        model: complied model ready for training
        train_dataset, val_dataset: preprocssed training and validation dataset
    Return:
        acc: the training accuracy as a float number
    """
    earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.001,
                                                patience = 5, mode = 'min', verbose = 1,
                                                restore_best_weights = True)
    re_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                                                patience = 2, min_delta = 0.001,
                                                mode = 'min', verbose = 1)

    # Train the model
    history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks = [earlystop,re_lr])

    acc = 0
    for ele in history.history['accuracy']:
        if ele > acc:
            acc = ele

    return acc


def evaluate_CNN_LSTM(model, test_dataset, test):  
    """
    This function evaluate the performance of the CNN or LSTM model.
    Args:
        model: trained model
        test_dataset: preprocssed test dataset
        test: the original test dataframe
    Return:
        accuracy: the testing accuracy as a float number
    """

    test_labels = tf.keras.utils.to_categorical(test["discourse_effectiveness"], num_classes=3)

    # Use model.predict() to get the predicted probabilities for each class
    y_pred = model.predict(test_dataset)

    # Convert the predicted probabilities to class labels
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Compute the accuracy of the model
    accuracy = accuracy_score(test["discourse_effectiveness"], y_pred_labels)
    #print('Test Accuracy:', accuracy)

    # Compute the precision and recall of the model
    precision = precision_score(test["discourse_effectiveness"], y_pred_labels, average='weighted')
    recall = recall_score(test["discourse_effectiveness"], y_pred_labels, average='weighted')
    #print('Test Precision:', precision)
    #print('Test Recall:', recall)

    # Compute the log loss of the model
    log_loss = np.mean(tf.keras.losses.categorical_crossentropy(test_labels, y_pred))
    #print('Test Log Loss:', log_loss)
    return accuracy


def createLSTM(train, val, test):
    """
    This function preprocss the data and create the LSTM model for classfication.
    Args:
        train: the training set in dataframe
        val : the validation set in dataframe
        test: the testing set in dataframe
    Retrun:
        model: complied model ready for training
        train_dataset, val_dataset, test_dataset: preprocssed dataset for training and testing
    """
    #LSTM
    #acc: 0.6415556159912973 precision: 0.6104462871047702 recall: 0.6415556159912973 loss: 0.8147903
    # Tokenize the text and convert to sequences
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(train["discourse_text"])
    train_sequences = tokenizer.texts_to_sequences(train["discourse_text"])
    val_sequences = tokenizer.texts_to_sequences(val["discourse_text"])
    test_sequences = tokenizer.texts_to_sequences(test["discourse_text"])

    # Pad the sequences
    max_len = 200
    train_data = pad_sequences(train_sequences, maxlen=max_len)
    val_data = pad_sequences(val_sequences, maxlen=max_len)
    test_data = pad_sequences(test_sequences, maxlen=max_len)

    # Convert the labels to one-hot encoded format
    train_labels = tf.keras.utils.to_categorical(train["discourse_effectiveness"], num_classes=3)
    val_labels = tf.keras.utils.to_categorical(val["discourse_effectiveness"], num_classes=3)
    test_labels = tf.keras.utils.to_categorical(test["discourse_effectiveness"], num_classes=3)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(len(train_data)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)
    
    # Define the LSTM model architecture
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=200, input_length=200))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model, train_dataset, val_dataset, test_dataset


def trainLSTM(model, train_dataset, val_dataset):
    """
    This function trains the created LSTM model.
    Args:
        model: complied model ready for training
        train_dataset, val_dataset: preprocssed training and validation dataset
    Return:
        acc: the training accuracy as a float number
    """

    earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.001,
                                                patience = 5, mode = 'min', verbose = 1,
                                                restore_best_weights = True)
    re_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                                                patience = 2, min_delta = 0.001,
                                                mode = 'min', verbose = 1)

    history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks = [earlystop,re_lr])

    acc = 0
    for ele in history.history['accuracy']:
        if ele > acc:
            acc = ele

    return acc

def createEnsemble(train, val, test):
    """
    This function preprocss the data and create the ensemble model for classfication.
    Args:
        train: the training set in dataframe
        val : the validation set in dataframe
        test: the testing set in dataframe
    Retrun:
        model: complied model ready for training
        train_dataset, val_dataset, test_dataset: preprocssed dataset for training and testing
    """
    #ensemble
    # loss: 0.4975 - accuracy: 0.8074 - precision: 0.8174 - recall: 0.7920
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    one_hot = OneHotEncoder()
    nlp_spacy = spacy.load('en_core_web_sm')
    count_ = CountVectorizer()

    stop_words = stopwords.words('english')
    ps = PorterStemmer()

    def text_preprocessing_new(sentence):
        """
        This function is used to preprocess the text, including remove punctuation, stop words and numeric values.
        Args : 
            sentence: the text need to be preprocssed
        Return: 
            result: the refined setence 
        """
        cln_sent = []
        for token_ in sentence.split(' '):
            if token_ in stop_words:
                pass
            else:
                cln_sent.append(ps.stem(token_))
        sen = ' '.join(cln_sent)
        result = re.sub('\W',' ', sen)
        result = re.sub('([0-9])+[a-z]+','', result)
        result = re.sub('\W{2,}','', result)
        return result
    
    train_data = pd.concat([train,val])
    train_data['clean_text'] = train_data.discourse_text.apply(text_preprocessing_new)
    datafor_0 = train_data.loc[train_data.discourse_effectiveness==0,:]
    datafor_1 = train_data.loc[train_data.discourse_effectiveness==1,:]
    datafor_2 = train_data.loc[train_data.discourse_effectiveness==2,:]

    train_data_balanced = pd.concat([datafor_1,datafor_2,datafor_2,datafor_0,datafor_0,datafor_0], axis=0)
    train_data_balanced.reset_index(inplace=True)
    train_data_balanced.drop('index', axis=1, inplace=True)
    train_data_balanced['word_count'] = train_data_balanced.clean_text.apply(lambda x: len(x.split(' ')))
    new_dataset = train_data_balanced.loc[train_data_balanced.word_count < 50,:]
    position_X = one_hot.fit_transform(new_dataset[['discourse_type']])
    position = pd.DataFrame(position_X.todense())
    vocabulary_training = 55000
    length_of_sentence = 30
    text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=vocabulary_training,
                                                   output_sequence_length=length_of_sentence,
                                                   standardize='lower_and_strip_punctuation',
                                                   output_mode='int')
    text_vectorizer.adapt(new_dataset.clean_text)
    text_embedding = tf.keras.layers.Embedding(mask_zero=True,
                         input_dim=vocabulary_training,
                         output_dim=128)
    Y = tf.keras.utils.to_categorical(new_dataset.discourse_effectiveness)
    X = new_dataset.clean_text
    x_train, x_test, y_train, y_test= train_test_split(X, Y)
    pos_train, pos_test, pos_ytrain, pos_ytest= train_test_split(position, Y)
    training_x = tf.data.Dataset.from_tensor_slices((x_train,pos_train))
    testing_x = tf.data.Dataset.from_tensor_slices((x_test,pos_test))
    training_output_data = tf.data.Dataset.from_tensor_slices((y_train))
    testing_output_data = tf.data.Dataset.from_tensor_slices((y_test))
    train_dataset = tf.data.Dataset.zip((training_x, training_output_data))
    val_dataset = tf.data.Dataset.zip((testing_x, testing_output_data))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    test['clean_txt'] = test.discourse_text.apply(text_preprocessing_new)
    test['word_count'] = test.clean_txt.apply(lambda x : len(x.split(' ')))
    test.loc[test.word_count < 50, ['word_count']].count()
    pos_test = one_hot.transform(test[['discourse_type']]).toarray()

    input_layer = layers.Input(shape=(1, ), dtype=tf.string, ragged=False)
    vectorizer_layer = text_vectorizer(input_layer)
    embedding_layer = text_embedding(vectorizer_layer)
    lstm_layer = layers.LSTM(23, return_sequences=True)(embedding_layer)
    conv_layer = layers.Conv1D(64, kernel_size=3, activation='relu', name='convolution_layer')(lstm_layer)
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)
    normalization_layer = layers.BatchNormalization()(pooling_layer)
    dense_layer = layers.Dense(500, activation='relu')(normalization_layer)
    drop_out = layers.Dropout(0.3)(dense_layer)
    dense_layer_level1 = layers.Dense(124, activation='relu')(drop_out)
    text_model = tf.keras.Model(input_layer, dense_layer_level1)

    position_input_layer = layers.Input(shape=(7,))
    ps_dense = layers.Dense(50, activation='relu')(position_input_layer)
    drop_lev1 = layers.Dropout(0.2)(ps_dense)
    ps_dense_2 = layers.Dense(5, activation='relu')(drop_lev1)
    position_model = tf.keras.Model(position_input_layer, ps_dense_2)

    #Combine
    layer_add = layers.Concatenate(name='concatinating_hybrid', axis=1)([text_model.output,position_model.output])
    combined_dropout = layers.Dropout(0.3)(layer_add)
    combined_dense = layers.Dense(128, activation='relu')(combined_dropout)
    output_layer = layers.Dense(3, activation='softmax')(combined_dense)


    ensembled_model = tf.keras.Model(inputs=[text_model.input, position_model.input],
                                    outputs = output_layer)
    
    ensembled_model.compile(loss=tf.keras.losses.categorical_crossentropy,
               optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
               metrics=['accuracy'])

    return ensembled_model, train_dataset, val_dataset, pos_test
    
def trainEnsemble(model, train_dataset, val_dataset):
    """
    This function trains the created ensemble model.
    Args:
        model: complied model ready for training
        train_dataset, val_dataset: preprocssed training and validation dataset
    Return:
        acc: the training accuracy as a float number
    """
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 5, mode = 'min', restore_best_weights = True)
    history = model.fit(train_dataset,
                        epochs=30,
                        validation_data = val_dataset,
                        callbacks=[earlyStopping]
                        )
    acc = 0
    for ele in history.history['accuracy']:
        if ele > acc:
            acc = ele
    return acc
    
    
def evaluateEnsemble(model, test_dataset, test):
    """
    This function evaluate the performance of the ensemble model.
    Args:
        model: trained model
        test_dataset: preprocssed test dataset
        test: the original test dataframe
    Return:
        accuracy: the testing accuracy as a float number
    """
    pred_proba = model.predict((test['clean_txt'],test_dataset),verbose=1)

    # Convert the predicted probabilities to class labels
    y_pred_labels = np.argmax(pred_proba, axis=1)

    # Compute the accuracy of the model
    accuracy = accuracy_score(test["discourse_effectiveness"], y_pred_labels)
    return accuracy


def seed_all(seed_value):
    import torch
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
        
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def createFastai(train, val, test):
    """
    This function preprocss the data and create the transformer model for classfication.
    Args:
        train: the training set in dataframe
        val : the validation set in dataframe
        test: the testing set in dataframe
    Retrun:
        model: complied model ready for training
        databunch: preprocssed dataset for training and testing
    """

    #fastai（need to use specific version of fastai and transformers)
    #acc: 0.6369322817514278 precision: 0.6127414586106014 recall: 0.6369322817514278 loss: 0.81764
    #this model is implemented to run on GPU

    #!pip install fastai==1.0.58
    #!pip install transformers==2.5.1

    import torch.nn as nn 
    from fastai import all
    from fastai.text import all
    from fastai.callbacks import all

    # transformers
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
    from transformers import AdamW
    from functools import partial

    temp = pd.concat([train,val])

    MODEL_CLASSES = {
        'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    }

    # Parameters
    seed = 42
    use_fp16 = False
    bs = 16
    model_type = 'roberta'
    pretrained_model_name = 'roberta-base'

    model_class,tokenizer_class, config_class = MODEL_CLASSES[model_type]
    model_class.pretrained_model_archive_map.keys()
    seed_all(seed)
    class TransformersBaseTokenizer(BaseTokenizer):
        """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
        def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
            self._pretrained_tokenizer = pretrained_tokenizer
            self.max_seq_len = pretrained_tokenizer.max_len
            self.model_type = model_type

        def __call__(self, *args, **kwargs): 
            return self

        def tokenizer(self, t:str) -> List[str]:
            """Limits the maximum sequence length and add the spesial tokens"""
            CLS = self._pretrained_tokenizer.cls_token
            SEP = self._pretrained_tokenizer.sep_token
            if self.model_type in ['roberta']:
                tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
                tokens = [CLS] + tokens + [SEP]
            else:
                tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
                if self.model_type in ['xlnet']:
                    tokens = tokens + [SEP] +  [CLS]
                else:
                    tokens = [CLS] + tokens + [SEP]
            return tokens
        
    transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
    transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
    fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])

    class TransformersVocab(Vocab):
        def __init__(self, tokenizer: PreTrainedTokenizer):
            super(TransformersVocab, self).__init__(itos = [])
            self.tokenizer = tokenizer
        
        def numericalize(self, t:Collection[str]) -> List[int]:
            "Convert a list of tokens `t` to their ids."
            return self.tokenizer.convert_tokens_to_ids(t)
            #return self.tokenizer.encode(t)

        def textify(self, nums:Collection[int], sep=' ') -> List[str]:
            "Convert a list of `nums` to their tokens."
            nums = np.array(nums).tolist()
            return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)
        
        def __getstate__(self):
            return {'itos':self.itos, 'tokenizer':self.tokenizer}

        def __setstate__(self, state:dict):
            self.itos = state['itos']
            self.tokenizer = state['tokenizer']
            self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

    transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)
    numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)

    tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)

    transformer_processor = [tokenize_processor, numericalize_processor]

    pad_first = bool(model_type in ['xlnet'])
    pad_idx = transformer_tokenizer.pad_token_id

    databunch = (TextList.from_df(temp, cols='discourse_text', processor=transformer_processor)
             .split_by_rand_pct(0.1,seed=seed)
             .label_from_df(cols= 'discourse_effectiveness')
             .add_test(test)
             .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))

    # defining the model architecture 
    class CustomTransformerModel(nn.Module):
        def __init__(self, transformer_model: PreTrainedModel):
            super(CustomTransformerModel,self).__init__()
            self.transformer = transformer_model
            
        def forward(self, input_ids, attention_mask=None):
            attention_mask = (input_ids!=pad_idx).type(input_ids.type()) 
            
            logits = self.transformer(input_ids,
                                    attention_mask = attention_mask)[0]   
            return logits
        
    config = config_class.from_pretrained(pretrained_model_name)
    config.num_labels = 3
    config.use_bfloat16 = use_fp16

    transformer_model = model_class.from_pretrained(pretrained_model_name, config = config)
    custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)

    CustomAdamW = partial(AdamW, correct_bias=False)

    learner = Learner(databunch, 
                    custom_transformer_model, 
                    opt_func = CustomAdamW, 
                    metrics=[accuracy, error_rate])

    # Show graph of learner stats and metrics after each epoch.
    learner.callbacks.append(ShowGraph(learner))

    # Put learn in FP16 precision mode. --> Seems to not working
    if use_fp16: learner = learner.to_fp16()

    # For roberta-base
    list_layers = [learner.model.transformer.roberta.embeddings,
                learner.model.transformer.roberta.encoder.layer[0],
                learner.model.transformer.roberta.encoder.layer[1],
                learner.model.transformer.roberta.encoder.layer[2],
                learner.model.transformer.roberta.encoder.layer[3],
                learner.model.transformer.roberta.encoder.layer[4],
                learner.model.transformer.roberta.encoder.layer[5],
                learner.model.transformer.roberta.encoder.layer[6],
                learner.model.transformer.roberta.encoder.layer[7],
                learner.model.transformer.roberta.encoder.layer[8],
                learner.model.transformer.roberta.encoder.layer[9],
                learner.model.transformer.roberta.encoder.layer[10],
                learner.model.transformer.roberta.encoder.layer[11],
                learner.model.transformer.roberta.pooler]
    
    learner.split(list_layers)
    learner.save('untrain')
    seed_all(seed)
    learner.load('untrain')
    return learner, databunch

def trainFastai(learner):
    """
    This function trains the transformer model created.
    Args:
        learner: the learner created for the transformer model
    Return:
        The training accuracy as a float
    """
    seed = 42
    num_groups = len(learner.layer_groups)
    learner.freeze_to(-1)
    learner.lr_find()
    learner.recorder.plot(skip_end=10,suggestion=True)
    learner.fit_one_cycle(4,max_lr=1e-03,moms=(0.8,0.7))
    learner.save('first_cycle')
    seed_all(seed)
    learner.load('first_cycle')
    learner.freeze_to(-2)
    lr = 1e-3
    learner.fit_one_cycle(6, max_lr=slice(lr*0.95**num_groups, lr), moms=(0.8, 0.9))
    learner.save('second_cycle')
    seed_all(seed)
    learner.load('second_cycle')
    learner.freeze_to(-3)
    learner.fit_one_cycle(2, max_lr=slice(lr*0.95**num_groups, lr), moms=(0.8, 0.9))
    learner.save('third_cycle')
    seed_all(seed)
    learner.load('third_cycle')

    res = learner.validate()

    return res[1].item()

def evaluateFastai(learner, databunch, test):
    """
    This function evaluate the performance of the transfomer model
    Args:
        learner: trained transformer model
        databunch: preprocssed data
        test: the original test data
    Return:
        accuracy: the testing accuracy as a float
    """
    def get_preds_as_nparray(ds_type) -> np.ndarray:
        #the get_preds method does not yield the elements in order by default
        preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()
        sampler = [i for i in databunch.dl(ds_type).sampler]
        reverse_sampler = np.argsort(sampler)
        return preds[reverse_sampler, :]

    test_preds = get_preds_as_nparray(DatasetType.Test)
    y_pred_labels = np.argmax(test_preds, axis=1)
    accuracy = accuracy_score(test["discourse_effectiveness"], y_pred_labels)
    return accuracy


def createTransformer(train, val, test):
    """
    This function preprocss the data and create the BERT model for classfication.
    Args:
        train: the training set in dataframe
        val : the validation set in dataframe
        test: the testing set in dataframe
    Retrun:
        m: complied model ready for training
        data_train, data_valid, dataset_test: preprocssed dataset for training and testing
    """

    #The transformer model implemented with tensorflow; required a CUDA enviroment to run on GPU
    #best performance model with lowest speed
    #acc: 0.9166 precision: 0.9195 recall: 0.9146 loss: 0.2124 
    
    from transformers import TFBertModel
    from transformers import AutoTokenizer

    temp = pd.concat([train,val])
    discourse_text = temp['discourse_text']
    discourse_type = temp['discourse_type']
    labels = temp['discourse_effectiveness']

    discourse_text_test = test['discourse_text']
    discourse_type_test = test['discourse_type']
    labels_test = test['discourse_effectiveness']

    discourse_text = discourse_text.str.strip()
    discourse_text_test = discourse_text_test.str.strip()

    def CleanFeatures(sentences):
        sentences = sentences.apply(lambda sequence:
                                                    [ltrs for ltrs in sequence if ltrs not in string.punctuation])
        sentences = sentences.apply(lambda wrd: ''.join(wrd))
        return sentences

    discourse_text = CleanFeatures(discourse_text)
    discourse_text_test = CleanFeatures(discourse_text_test)

    text = np.asarray([discourse_type[discourse_text.index[index]] + " "+ value for index, value in enumerate(discourse_text) ])
    text_test = np.asarray([discourse_type_test[discourse_text_test.index[index]] + " "+ value for index, value in enumerate(discourse_text_test) ])

    sequence_length = [len(str(te).split()) for te in text]
    SEQ_LEN = np.max(sequence_length)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    input_ids = []
    attention_mask = []
    input_ids_test = []
    attention_mask_test = []

    for index, value in enumerate(text):
        tokens = tokenizer.encode_plus(value, max_length= SEQ_LEN, padding = "max_length",
                                        truncation = True, return_token_type_ids = False,
                                        return_attention_mask = True, return_tensors = 'np')
        input_ids.append(tokens['input_ids'])
        attention_mask.append(tokens['attention_mask'])
    for index, value in enumerate(text_test):
        tokens = tokenizer.encode_plus(value, max_length= SEQ_LEN, padding = "max_length",
                                        truncation = True, return_token_type_ids = False,
                                        return_attention_mask = True, return_tensors = 'np')
        input_ids_test.append(tokens['input_ids'])
        attention_mask_test.append(tokens['attention_mask'])

    input_ids = np.asarray(input_ids)
    attention_mask = np.asarray(attention_mask)

    input_ids_test = np.asarray(input_ids_test)
    attention_mask_test = np.asarray(attention_mask_test)

    input_ids = input_ids.reshape(input_ids.shape[0], input_ids.shape[2])
    attention_mask = attention_mask.reshape(attention_mask.shape[0], attention_mask.shape[2])

    input_ids_test = input_ids_test.reshape(input_ids_test.shape[0], input_ids_test.shape[2])
    attention_mask_test = attention_mask_test.reshape(attention_mask_test.shape[0], attention_mask_test.shape[2])

    label_ = LabelEncoder()
    labels = label_.fit_transform(labels)
    labels = to_categorical(labels)
    classes_names = list(label_.classes_)
    dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_mask, labels))

    labels_test = to_categorical(labels_test)
    dataset_test = tf.data.Dataset.from_tensor_slices((input_ids_test, attention_mask_test, labels_test))
    def map_change_dataset(input_ids, attention_mask, labels):
        return {'input_ids':input_ids, 'attention_mask': attention_mask}, labels
    
    dataset = dataset.map(map_change_dataset)
    dataset = dataset.shuffle(500000).batch(84)

    dataset_test = dataset_test.map(map_change_dataset)
    dataset_test = dataset_test.shuffle(500000).batch(84)

    training_split = 0.88
    dataset_len = len(list(dataset))
    data_train = dataset.take(round(training_split*dataset_len))
    data_valid = dataset.skip(round(training_split*dataset_len))

    bert = TFBertModel.from_pretrained('bert-base-cased')
    input_ids_m = tf.keras.layers.Input(shape = (SEQ_LEN, ), name = "input_ids", dtype = 'int32')
    attention_mask_n = tf.keras.layers.Input(shape = (SEQ_LEN, ), name = "attention_mask", dtype = 'int32')
    bert_m = bert(input_ids_m, attention_mask = attention_mask_n)[0]
    x = tf.keras.layers.LSTM(128, return_sequences= True)(bert_m)
    x1 = tf.keras.layers.GlobalAveragePooling1D()(x)
    x2 = tf.keras.layers.GlobalMaxPool1D()(x)
    x = tf.keras.layers.concatenate([x1, x2])
    x = tf.keras.layers.Dense(128, activation = "relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    y = tf.keras.layers.Dense(len(classes_names), activation = "softmax")(x)

    m = tf.keras.models.Model(inputs = [input_ids_m, attention_mask_n], outputs = y)

    m.layers[2].trainable = False

    m.compile(loss="categorical_crossentropy",optimizer= "adam",
              metrics=["accuracy",
                       tf.keras.metrics.Precision(), 
                       tf.keras.metrics.Recall()])
    
    return m, data_train, data_valid, dataset_test

def trainTransformer(m, data_train, data_valid):
    """
    The function trains the created BERT model.
    Args:
        m: complied model ready for training
        train_dataset, val_dataset: preprocssed training and validation dataset
    Return:
        acc: the training accuracy as a float number
    """
    earlyStopping =  tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 5, mode = 'min', restore_best_weights = True)
    history = m.fit(data_train,
                            epochs= 40,
                            use_multiprocessing=True,
                            validation_data = data_valid,
                            callbacks=[earlyStopping])
    
    acc = 0
    for ele in history.history['accuracy']:
        if ele > acc:
            acc = ele
    return acc

def evaluateTransformer(m, dataset_test, test):
    """
    This function evaluate the performance of the BERT model.
    Args:
        model: trained model
        test_dataset: preprocssed test dataset
        test: the original test dataframe
    Return:
        acc: the testing accuracy as a float number
    """
    result = m.evaluate(dataset_test)
    acc = result[1]
    return acc

