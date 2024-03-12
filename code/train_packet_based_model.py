import tensorflow as tf  
import numpy as np 
import os 
import sys 
import time 
from tensorflow.keras.layers import Dense, Flatten  
from tensorflow.keras import Model  


class PacketModel(Model):  # Define a classe PacketModel que herda da classe Model.
    def __init__(self):  # Define o método construtor da classe PacketModel.
        super(PacketModel, self).__init__()  # Chama o construtor da classe pai.
        self.flatten = Flatten()  # Inicializa a camada de achatamento.
        self.d1 = Dense(2048, activation='relu')  # Inicializa a primeira camada densa com ativação ReLU.
        self.d2 = Dense(num_input*timesteps, activation='relu')  # Inicializa a segunda camada densa com ativação ReLU.

    def call(self, x):  # Define o método de chamada da classe PacketModel.
        x = self.flatten(x)  # Aplica a camada de achatamento.
        dense1_out = self.d1(x)  # Passa os dados pela primeira camada densa.
        dense2_out = self.d2(dense1_out)  # Passa os dados pela segunda camada densa.
        d2_out_clip = tf.clip_by_value(dense2_out, 0., 1.)  # Aplica a função de clip aos dados.
        output = tf.reshape(d2_out_clip, (-1, timesteps, num_input))  # Realiza o redimensionamento dos dados de saída.
        return output


def get_files(day, prefix='../data/packet_based/'):  # Define a função para obter os arquivos de dados.
    all_files = []  # Inicializa uma lista para armazenar todos os arquivos.
    prefix = prefix + day  # Concatena o prefixo com o dia da semana.
    for file in os.listdir(prefix):  # Itera sobre os arquivos no diretório especificado.
        if file.endswith(".npy") and file.startswith('part'):  # Verifica se o arquivo é do tipo .npy e tem o prefixo 'part'.
            all_files.append(os.path.join(prefix, file))  # Adiciona o caminho completo do arquivo à lista.
    all_files = sorted(all_files)  # Ordena os arquivos na lista.
    return all_files  # Retorna a lista de arquivos.


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


def get_train_ds():  # Define a função para obter o conjunto de dados de treinamento.
    train_files = get_files('monday', prefix='../data/packet_based/')  # Obtém os arquivos de dados para treinamento.
    x_train = []  # Inicializa uma lista para armazenar os dados de treinamento.
    for f in train_files:  # Itera sobre os arquivos de treinamento.
        print(f)  # Imprime o nome do arquivo.
        x_train.append(np.load(f))  # Carrega os dados do arquivo e os adiciona à lista.
    x_train = np.concatenate(x_train, axis=0)  # Concatena os dados de todos os arquivos em um único conjunto.
    x_train = x_train.astype(np.float32)  # Converte os dados para o tipo float32.
    x_train_min = np.min(x_train, axis=0)  # Calcula o valor mínimo ao longo das colunas.
    x_train_max = np.max(x_train, axis=0)  # Calcula o valor máximo ao longo das colunas.
    x_train_normalized = (x_train - x_train_min) / (x_train_max - x_train_min + 0.000001)  # Normaliza os dados de treinamento.
    return x_train, x_train_normalized, x_train_min, x_train_max  # Retorna os conjuntos de dados de treinamento.


def train_model(x_train, x_train_normalized, timesteps, num_input):  # Define a função para treinar o modelo.
    num_iters = 10000  # Define o número de iterações de treinamento.
    batch_size = 512  # Define o tamanho do lote de treinamento.
    last_valid_index = len(x_train_normalized) - timesteps - 1  # Calcula o último índice válido para criar lotes de dados.
    start_time = time.time()  # Registra o tempo inicial do treinamento.
    for learning_rate in [0.001, 0.0001, 0.00001]:  # Itera sobre diferentes taxas de aprendizado.
        x_train_partial = make_partial(x_train_normalized, 0.75)  # Cria conjuntos de dados parcialmente mascarados.
        print('x_train_partial', x_train_partial.dtype)  # Imprime o tipo de dados do conjunto de dados parcial.
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # Inicializa o otimizador Adam com a taxa de aprendizado atual.
        grads = []  # Inicializa uma lista para armazenar os gradientes.
        for v in model.trainable_variables:  # Itera sobre os parâmetros treináveis do modelo.
            grads.append(np.zeros(v.shape))  # Adiciona gradientes nulos à lista.
        optimizer.apply_gradients(zip(grads, model.trainable_variables))  # Aplica os gradientes nulos para inicializar os parâmetros.
        train_loss.reset_states()  # Reinicia o estado da perda de treinamento.
        for n in range(num_iters):  # Itera sobre as iterações de treinamento.
            x_batch = np.zeros((batch_size, timesteps, num_input), dtype=np.float32)  # Inicializa um lote de entrada com zeros.
            y_batch = np.zeros((batch_size, timesteps, num_input), dtype=np.float32)  # Inicializa um lote de saída com zeros.
            jj = 0  # Inicializa o índice para o lote atual.
            while jj < len(x_batch):  # Loop até o lote atual estar completo.
                r = np.random.randint(0, last_valid_index)  # Gera um índice aleatório dentro dos limites válidos.
                if np.sum(x_train[r + timesteps - 1]) == 0:  # Verifica se os dados contêm apenas zeros.
                    continue  # Se sim, passa para a próxima iteração.
                x_batch[jj] = x_train_partial[r:r + timesteps]  # Define os dados de entrada do lote atual.
                y_batch[jj] = x_train_normalized[r:r + timesteps]  # Define os dados de saída do lote atual.
                jj += 1  # Incrementa o índice do lote atual.
            train_step(x_batch, y_batch, optimizer)  # Executa um passo de treinamento.
            if n % 1000 == 0:  # Verifica se é hora de imprimir o progresso do treinamento.
                print('trained time', time.time() - start_time, n, 'out of', num_iters, 'loss:', train_loss.result().numpy())  # Imprime o tempo decorrido, o número de iterações e a perda atual.
    print('total time', time.time() - start_time)  # Imprime o tempo total decorrido durante o treinamento.


if __name__ == "__main__":  # Verifica se o script está sendo executado como o programa principal.
    x_train, x_train_normalized, _, _ = get_train_ds()  # Obtém os conjuntos de dados de treinamento.
    print('Train set shape and type:', x_train_normalized.shape, x_train_normalized.dtype)  # Imprime o formato e o tipo dos dados de treinamento.
    timesteps = 20  # Define o número de passos no tempo.
    num_input = 29  # Define o número de entradas.
    model = PacketModel()  # Inicializa o modelo PacketModel.
    model._set_inputs(tf.TensorSpec([None, timesteps, num_input]))  # Define as especificações de entrada do modelo.
    train_loss = tf.keras.metrics.Mean(name='train_loss')  # Inicializa a métrica de perda de treinamento.
    train_model(x_train, x_train_normalized, timesteps, num_input)  # Treina o modelo.
    model.save('../models/packet_based_model')  # Salva o modelo treinado.
