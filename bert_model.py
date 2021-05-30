import bert
import os

from transformers import AutoTokenizer
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow.keras as keras
from tensorflow.python.keras.layers import InputLayer,Lambda,Dense,Dropout
import pickle


class Models:
    def tokenizeData(self,data,max_length):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        tokens = tokenizer.batch_encode_plus(
            data,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf')
        return np.array(tokens['input_ids'])
    def labelBinarizer(self,labels):
        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)
        pickle.dump(lb,open('LabelBinarizer.pkl', 'wb'))
        return labels,lb.classes_

class AlbertModel(Models):
    def __init__(self):
        model_name = "albert_tiny"
        model_dir = bert.fetch_brightmart_albert_model(model_name, ".models")
        self.model_ckpt = os.path.join(model_dir, "albert_model.ckpt")
        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        self.l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        self.l_bert.trainable = True

        self.max_length = 500

    def fit(self,contents,labels):

        contents = self.tokenizeData(contents,self.max_length)
        labels,classes = self.labelBinarizer(labels)

        x_train,x_test,y_train,y_test = train_test_split(contents,labels,test_size=0.2,random_state=1234,shuffle=True)


        model = keras.models.Sequential()
        model.add(InputLayer(input_shape=self.max_length,))
        model.add(self.l_bert)
        model.add(Lambda(lambda x:x[:,0,:]))
        model.add(Dense(128,activation='relu'))
        model.add(Dense(len(classes),activation='sigmoid'))

        model.build(input_shape=(None, self.max_length))
        bert.load_albert_weights(self.l_bert, self.model_ckpt)

        model.compile(loss=keras.losses.categorical_crossentropy,optimizer = keras.optimizers.Adam(),metrics = ['accuracy'])
        model.fit(x = x_train,y = y_train , validation_data=(x_test,y_test),epochs=5,batch_size=32,verbose=1)

        model.save('crime')

    def load(self):
        self.model = keras.models.load_model('crime')

    def predict(self,contents):
        lb = pickle.load(open('LabelBinarizer.pkl', 'rb'))
        print(lb.classes_)
        self.load()
        contents = self.tokenizeData(contents,self.max_length)
        return lb.classes_,self.model.predict(contents)
