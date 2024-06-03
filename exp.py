# import json
# import os

# file = open('intents-faq.json')
# data = json.load(file)
# # print(data['intents'])

# # os.system('cls')
# for i in range(len(data["intents"])):
#     print(data['intents'][i]['patterns'], "\n")

import pandas as pd
df = pd.read_csv('./data/intents.csv', names=["pattern", 'tag', 'response'])
print(df["response"][0])