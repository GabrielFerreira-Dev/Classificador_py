from pickle import load
import numpy as np

# Inferência de uma nova instância
nova_instancia = [[2010, 3.968, 9.122, 0.821, 55, 0.529, -0.020, 8, 0.489, 0.246]]

# Normalizar a nova instância
normalizador = load(open('C:\\Users\\gabri\\PycharmProjects\\Clusterizacao_1\\normalizador\\normalizador_happiness.pkl', 'rb'))
nova_instancia_normalizada = normalizador.transform(nova_instancia)

# Classificar a nova instância
happiness_classificador = load(open('C:\\Users\\gabri\\PycharmProjects\\Clusterizacao_1\\normalizador\\happiness_tree_model_cross.pkl', 'rb'))

# Classificar
resultado = happiness_classificador.predict(nova_instancia_normalizada)
dist_proba = happiness_classificador.predict_proba(nova_instancia_normalizada)

indice = np.argmax(dist_proba[0])
classe_predita = happiness_classificador.classes_[indice]
score = dist_proba[0][indice]
print("Classificado como :", classe_predita, 'Score:', str(score))
print(np.argmax(dist_proba[0]))
print(happiness_classificador.classes_)