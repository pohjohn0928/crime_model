# -*- coding: utf-8 -*-

from bert_model import AlbertModel
from get_data import DataHelper

data = DataHelper()
albert = AlbertModel()
#
contents,labels = data.get_data()
albert.fit(contents,labels)

