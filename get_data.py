import json
import csv
from opencc import OpenCC


class DataHelper:
    def makeCSV(self):
        file = open('data_train.json',encoding='utf-8')
        facts = []
        accusations = []

        theft = 0
        harm = 0
        fraud = 0

        cc = OpenCC('s2t')
        for data in file:
            data = json.loads(data)

            fact = data['fact']
            crime = data['meta']['accusation'][0]

            if crime == '盗窃' or crime == '故意伤害' or crime == '诈骗':
                if crime == '盗窃' and theft < 1000:
                    theft += 1
                    facts.append(cc.convert(fact))
                    accusations.append(cc.convert(crime))
                elif crime == '故意伤害' and harm < 1000:
                    harm += 1
                    facts.append(cc.convert(fact))
                    accusations.append(cc.convert(crime))
                elif crime == '诈骗' and fraud < 1000:
                    fraud += 1
                    facts.append(cc.convert(fact))
                    accusations.append(cc.convert(crime))

        with open('crime.csv', 'w', newline='',encoding='utf-8') as csvfile:
            fieldnames = ['Fact', 'accusation']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(accusations)):
                writer.writerow({'Fact' : facts[i],'accusation' : accusations[i]})


    def get_data(self):
        csvfile = open('crime.csv',newline='',encoding='utf-8')
        reader = csv.DictReader(csvfile)
        facts = []
        accusations = []
        for row in reader:
            facts.append(row['Fact'])
            accusations.append(row['accusation'])
        return facts,accusations
