from distutils.command.config import config
from email.errors import StartBoundaryNotFoundDefect
import pandas as pd 
import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Simple İmputer yapacağız. Eksik veriler için.
yorumlar = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python-Machine-Learning/csvexamplefolder/Restaurant_Reviews.csv',error_bad_lines=False)


ps = PorterStemmer()
nltk.download('stopwords')

derlem = []
for i in range(716):    
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)
    # print(yorum)



cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(derlem).toarray() # bağımsız değişken
y = yorumlar.iloc[:,1].values # bağımlı değişken    


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)

sc = StandardScaler()

gnb = GaussianNB()
result = gnb.fit(X_train,y_train)

gnb_y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,gnb_y_pred)


# print(gnb_y_pred)
# print(cm)
# print(X)
print(y)
# print(stop)
# print(yorum)
# print(yorumlar)

