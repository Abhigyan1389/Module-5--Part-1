import pandas as pd
#load dataset
df = pd.read_csv('data.csv)
                 #calculate count,mean,std,min,quartiles, and max
                 summary = df.describe()
                 print (summary)
