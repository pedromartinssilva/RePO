import tensorflow as tf  
import numpy as np  
import os  
import sys 
import time
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import Model
import pandas as pd


class FlowModel(Model):  # Define a classe FlowModel que herda da classe Model.
    def __init__(self, input_size):  # Define o método construtor da classe FlowModel.
        super(FlowModel, self).__init__()  # Chama o construtor da classe pai.
        self.flatten = Flatten()  # Inicializa a camada de achatamento.
        self.d1 = Dense(256, activation='relu')  # Inicializa a primeira camada densa com ativação ReLU.
        self.d2 = Dense(256, activation='relu')  # Inicializa a segunda camada densa com ativação ReLU.
        self.d3 = Dense(256, activation='relu')  # Inicializa a terceira camada densa com ativação ReLU.
        self.drop1 = Dropout(0.5)  # Inicializa a camada de dropout.
        self.d4 = Dense(256, activation='relu')  # Inicializa a quarta camada densa com ativação ReLU.
        self.d5 = Dense(256, activation='relu')  # Inicializa a quinta camada densa com ativação ReLU.
        self.drop2 = Dropout(0.5)  # Inicializa a segunda camada de dropout.
        self.d6 = Dense(256, activation='relu')  # Inicializa a sexta camada densa com ativação ReLU.
        self.d7 = Dense(input_size, activation='relu')  # Inicializa a sétima camada densa com ativação ReLU.

    def call(self, x, training=None):  # Define o método de chamada da classe FlowModel.
        # Passa os dados pelas camadas densas e aplica as camadas de dropout.
        x = self.d1(x)  
        x = self.d2(x)  
        x = self.d3(x)  
        x = self.drop1(x, training=training) 
        x = self.d4(x)  
        x = self.d5(x) 
        x = self.drop2(x, training=training) 
        x = self.d6(x)
        x = self.d7(x) 
        x = tf.clip_by_value(x, 0., 1.)  # Aplica a função de clip aos dados.
        return x 


def get_train_ds():  # Define a função para obter o conjunto de dados de treinamento.
    url_data = '../data/flow_based/Monday-WH-generate-labeled.csv'  # Define o caminho para o arquivo CSV de dados.
    df = pd.read_csv(url_data)  # Lê o arquivo CSV e carrega os dados em um DataFrame.
    feats = df.iloc[:, 8:]  # Seleciona as características (features) dos dados.
    ds_port = df.iloc[:, 5]  # Seleciona a porta de serviço dos dados.
    df = pd.concat([ds_port, feats], axis=1)  # Concatena a porta de serviço com as características.
    print(df.columns.values)  # Imprime os nomes das colunas do DataFrame.
    all_feats = df.iloc[:, :-1].astype(np.float32).values  # Converte as características para um array numpy.
    known_data_IDs = (np.any(np.isinf(all_feats), axis=1) + np.any(np.isnan(all_feats), axis=1)) == False  # Identifica os IDs dos dados conhecidos.
    x_train = all_feats[known_data_IDs]  # Seleciona os dados de entrada conhecidos.

    y_train = df.iloc[:, -1].values  # Seleciona as etiquetas dos dados.
    y_train[y_train == 'BENIGN'] = 0.  # Substitui 'BENIGN' por 0 nas etiquetas.
    y_train = y_train.astype(np.float32)  # Converte as etiquetas para o tipo float32.
    y_train = y_train[known_data_IDs]  # Seleciona as etiquetas dos dados conhecidos.

    print(x_train.shape, y_train.shape)  # Imprime o formato dos dados de entrada e das etiquetas.
    
    # Calcula o valor mínimo ao longo das colunas dos dados de entrada.
    train_min = np.min(x_train, axis=0) 
    train_max = np.max(x_train, axis=0)  

    x_train = (x_train - train_min) / (train_max - train_min + 1e-6)  # Normaliza os dados de entrada.
    
    return x_train  # Retorna os dados de entrada normalizados.


def make_partial(dset, masked_percent=0.75):  # Define a função para criar um conjunto de dados parcialmente mascarado.
    partial = dset  # Define o conjunto de dados parcial como o conjunto de dados original.
    mask = np.random.random(partial.shape)  # Gera uma máscara aleatória do mesmo formato do conjunto de dados.
    mask = (mask > masked_percent) * 1.  # Aplica uma máscara binária aos valores da máscara.
    mask = mask.astype(np.float32)  # Converte a máscara para o tipo float32.
    partial = partial * mask  # Aplica a máscara ao conjunto de dados parcial.
    return partial  # Retorna o conjunto de dados parcialmente mascarado.


@tf.function  # Anota a função para compilação em grafo pelo TensorFlow.
def repo_loss(rec_x, x):  # Define a função de perda para cálculo do erro médio quadrático.
    MSE_loss = (rec_x - x)**2  # Calcula o erro quadrático entre os dados reconstruídos e os dados originais.
    MSE_loss = tf.reduce_mean(MSE_loss)  # Calcula a média do erro quadrático.
    return MSE_loss  # Retorna o erro médio quadrático.


@tf.function  # Anota a função para compilação em grafo pelo TensorFlow.
def train_step(partial_x, x, optimizer):  # Define a função para um passo de treinamento.
    with tf.GradientTape() as tape:  # Inicia um contexto de gravação de gradientes.
        rec_x = model(partial_x, training=True)  # Obtém a reconstrução dos dados de entrada.
        loss = repo_loss(rec_x=rec_x, x=x)  # Calcula a perda com base na reconstrução e nos dados originais.
    gradients = tape.gradient(loss, model.trainable_variables)  # Calcula os gradientes da perda em relação aos parâmetros do modelo.
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Aplica os gradientes para atualizar os parâmetros do modelo.
    train_loss(loss)  # Registra a perda do treinamento.


def train_model(x_train):  # Define a função para treinar o modelo.
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Inicializa o otimizador Adam com a taxa de aprendizado especificada.
    nb_epochs = 5  # Define o número de épocas de treinamento.
    batch_size = 256  # Define o tamanho do lote de treinamento.
    total_batch = len(x_train) // batch_size  # Calcula o número total de lotes.
    if len(x_train) % batch_size != 0:  # Verifica se há um lote incompleto.
        total_batch += 1  # Incrementa o número total de lotes.

    start_time = time.time()  # Registra o tempo inicial do treinamento.
    for ep in range(nb_epochs):  # Itera sobre as épocas de treinamento.
        train_loss.reset_states()  # Reinicia o estado da perda de treinamento.
        x_train_partial = make_partial(x_train, 0.75)  # Cria conjuntos de dados parcialmente mascarados.
        inds = rng.permutation(x_train_partial.shape[0])  # Permuta os índices dos dados parcialmente mascarados.
        x_train_partial_perm = x_train_partial[inds]  # Aplica a permutação aos dados parcialmente mascarados.
        x_train_perm = x_train[inds]  # Aplica a permutação aos dados de entrada.
        for i in range(total_batch):  # Itera sobre os lotes de treinamento.
            x_batch = x_train_partial_perm[i * batch_size:(i + 1) * batch_size]  # Seleciona o lote parcialmente mascarado.
            y_batch = x_train_perm[i * batch_size:(i + 1) * batch_size]  # Seleciona o lote de entrada correspondente.
            train_step(x_batch, y_batch, optimizer)  # Executa um passo de treinamento.
        print('trained time', time.time() - start_time, ep, 'loss:', train_loss.result().numpy())  # Imprime o tempo decorrido e a perda atual.
    print('total time', time.time() - start_time)  # Imprime o tempo total decorrido durante o treinamento.


if __name__ == "__main__":  # Verifica se o script está sendo executado como o programa principal.
    RANDOM_SEED = 2019  # Define a semente aleatória para reproducibilidade.
    rng = np.random.RandomState(RANDOM_SEED)  
    x_train = get_train_ds()  # Obtém os dados de treinamento.
    input_size = x_train.shape[1]  # Obtém o tamanho da entrada dos dados.

    model = FlowModel(input_size) 
    model._set_inputs(tf.TensorSpec([None, input_size]))  # Define as especificações de entrada do modelo.

    train_loss = tf.keras.metrics.Mean(name='train_loss')  # Inicializa a métrica de perda de treinamento.
    train_model(x_train)  

    model.save('../models/flow_based_model') 
