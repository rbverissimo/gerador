import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Suprimir mensagens de aviso
warnings.filterwarnings("ignore")

# Ler os dados do arquivo CSV
data = pd.read_csv('C:\\Users\\Vine\\Desktop\\dados\\dados.csv', header=None, usecols=range(6))
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna(axis=1)
data.columns = ['Ball_1', 'Ball_2', 'Ball_3', 'Ball_4', 'Ball_5', 'Draw']

# Dividir os dados em conjunto de treino e teste
train_data = data.iloc[:1806, :]
test_data = data.iloc[1806:, :]

# Transformar a variável test_y em um DataFrame do pandas com uma coluna nomeada como "Draw"
test_y = pd.DataFrame(test_data['Draw'])

# Treinar o modelo de regressão logística
train_X = train_data.iloc[:, :-1]
train_y = train_data.iloc[:, -1]
model = LogisticRegression(max_iter=10000)
model.fit(train_X, train_y)

# Realizar a classificação dos dados de teste
test_X = test_data.iloc[:, :-1]
predictions = model.predict(test_X)

# Calcular a acurácia e matriz de confusão
accuracy = accuracy_score(test_y, predictions)
confusion = confusion_matrix(test_y, predictions)

# Fazendo previsões para conjuntos de 6 dezenas
def predict_6_numbers(model, numbers):
    proba = model.predict_proba(numbers)
    sorted_indexes = proba.argsort()[::-1]
    predicted_numbers = []
    for i in range(6):
        for j in sorted_indexes[:,i]:
            if j+1 in range(20, 30) or j+1 in range(40, 61):
                predicted_numbers.append(j+1)
                break
    return predicted_numbers

# Gerando previsões para cada conjunto de 6 dezenas
n_predictions = 100
predicted_numbers = []
for i in range(len(test_X)):
    if i >= n_predictions:
        break
    test_numbers = test_X.iloc[[i]]
    predicted_numbers.append(predict_6_numbers(model, test_numbers))

# Imprimindo as previsões
print('Previsões de 6 dezenas:')
for i, numbers in enumerate(predicted_numbers):
    print(f'Jogo {i+1}: {", ".join(str(n) for n in numbers)}')
