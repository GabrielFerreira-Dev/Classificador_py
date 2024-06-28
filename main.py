import warnings
from collections import Counter
from pickle import dump

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Carregar os dados
dados = pd.read_csv('C:\\Users\\gabri\\PycharmProjects\\Clusterizacao_1\\dados\\World-happiness-report-updated_2024.csv', sep=',', encoding='latin1')

# Selecionar atributos e classes
dados_atributos = dados[['year', 'Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                         'Freedom to make life choices', 'Generosity', 'Perceptions of corruption', 'Positive affect', 'Negative affect']]
dados_classes = dados[['Country name']]

# Lidar com valores ausentes nos atributos
imputer = SimpleImputer(strategy='mean')
dados_atributos = imputer.fit_transform(dados_atributos)
dados_atributos = pd.DataFrame(dados_atributos, columns=['year', 'Life Ladder', 'Log GDP per capita', 'Social support',
                                                         'Healthy life expectancy at birth', 'Freedom to make life choices',
                                                         'Generosity', 'Perceptions of corruption', 'Positive affect',
                                                         'Negative affect'])

# Normalizar os dados
normalizador = preprocessing.MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_atributos)
dump(modelo_normalizador, open('C:\\Users\\gabri\\PycharmProjects\\Clusterizacao_1\\normalizador\\normalizador_happiness.pkl', 'wb'))

dados_atributos_normalizados = modelo_normalizador.transform(dados_atributos)
dados_atributos_normalizados = pd.DataFrame(dados_atributos_normalizados, columns=['year', 'Life Ladder', 'Log GDP per capita', 'Social support',
                                                                                   'Healthy life expectancy at birth', 'Freedom to make life choices',
                                                                                   'Generosity', 'Perceptions of corruption', 'Positive affect',
                                                                                   'Negative affect'])

# Combinar atributos normalizados com as classes
dados_finais = pd.concat([dados_atributos_normalizados, dados_classes.reset_index(drop=True)], axis=1)

# Verificar a contagem de amostras por classe
print('Frequência de classes antes do balanceamento:')
classes_count = Counter(dados_finais['Country name'])
print(classes_count)

# Remover classes com menos de 2 amostras
classes_remover = [classe for classe, count in classes_count.items() if count < 2]
dados_finais = dados_finais[~dados_finais['Country name'].isin(classes_remover)]

# Aplicar SMOTE
dados_atributos = dados_finais.drop(columns=['Country name'])
dados_classes = dados_finais['Country name']

# Construir um objeto SMOTE com k_neighbors ajustado
resampler = SMOTE(k_neighbors=1)

# Executar o balanceamento
dados_atributos_b, dados_classes_b = resampler.fit_resample(dados_atributos, dados_classes)

# Verificar a frequência das classes após o balanceamento
print('Frequência de classes após balanceamento:')
classes_count = Counter(dados_classes_b)
print(classes_count)

# Converter os dados balanceados em DataFrames
dados_atributos_b = pd.DataFrame(dados_atributos_b, columns=dados_atributos.columns)
dados_classes_b = pd.DataFrame(dados_classes_b, columns=['Country name'])

# AVALIAÇÃO DA ACURÁCIA COM CROSS VALIDATION E BUSCA PELOS MELHORES HIPERPARÂMETROS
# Parâmetros a serem testados
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Instanciar o classificador
tree = DecisionTreeClassifier()

# Realizar a busca pelos melhores hiperparâmetros
grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=10, scoring='accuracy')
grid_search.fit(dados_atributos_b, dados_classes_b)

# Melhor modelo encontrado
melhor_modelo = grid_search.best_estimator_

# Avaliar a acurácia com Cross-Validation
scoring = ['precision_macro', 'recall_macro']
scores_cross = cross_validate(melhor_modelo, dados_atributos_b, dados_classes_b, cv=10, scoring=scoring)

print(f'Melhores hiperparâmetros: {grid_search.best_params_}')
print(f'Precision média: {scores_cross["test_precision_macro"].mean()}')
print(f'Recall média: {scores_cross["test_recall_macro"].mean()}')

# Treinar o modelo com a base normalizada, balanceada e completa usando o melhor modelo encontrado
dados_tree = melhor_modelo.fit(dados_atributos_b, dados_classes_b)

# Salvar o modelo para uso posterior
dump(dados_tree, open('C:\\Users\\gabri\\PycharmProjects\\Clusterizacao_1\\normalizador\\happiness_tree_model_cross.pkl', 'wb'))


