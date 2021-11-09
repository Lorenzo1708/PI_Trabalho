"""
Discplina:
Processamento de Imagens (6498100)

Grupo:
Guilherme Schuffner Ribeiro - Matrícula: 655104
Henrique Dornas Mendes      - Matrícula: 651252
Lorenzo Duarte Costa        - Matrícula: 652641

Este Script deve ser executado antes de se executar o arquivo "ocr.py",
uma vez que ele precisa criar os arquivos dos modelos.

Para executá-lo, digite o seguinte comando no terminal:
python3 models.py
"""

import cv2
import PIL.Image
import PIL.ImageTk
import math
import time
import numpy
import scipy.interpolate
import joblib
import seaborn
import sklearn.svm
import sklearn.metrics
import tkinter
import tkinter.ttk
import tkinter.constants
import tkinter.filedialog
import matplotlib
import tensorflow

# Inicializar a interface.
root = tkinter.Tk()
root.title('MLP e SVM')
root.geometry('768x384')

# Inicializar um Widget de Scroll para a interface.
main_frame = tkinter.Frame(root)
main_frame.pack(expand=1, fill=tkinter.constants.BOTH)

canvas = tkinter.Canvas(main_frame)
canvas.pack(expand=1, fill=tkinter.constants.BOTH, side=tkinter.constants.LEFT)

scrollbar = tkinter.ttk.Scrollbar(main_frame, command=canvas.yview, orient=tkinter.constants.VERTICAL)
scrollbar.pack(fill=tkinter.constants.Y, side=tkinter.constants.RIGHT)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

second_frame = tkinter.Frame(canvas)

# Inicializar os Widgets da interface.
button_load_mnist = tkinter.Button(second_frame)
button_process_mnist = tkinter.Button(second_frame)
button_create_mlp = tkinter.Button(second_frame)
button_create_svm = tkinter.Button(second_frame)
label_time_elapsed = tkinter.Label(second_frame)
label_image = tkinter.Label(second_frame)

x_testing = []
x_training = []
y_testing = []
y_training = []
x_projection_testing = []
x_projection_training = []

# Função para criar e salvar o SVM.
# Para o SVM, utilizam-se todos os parâmetros padrões,
# exceto o fator de regularização C,
# o qual possui valor 12.
def create_svm() -> None:
    time_elapsed = time.time()

    svm = sklearn.svm.SVC(C=12)

    # Treinar o SVM.
    svm.fit(numpy.array(x_projection_training), y_training)
    
    # Declarar como global os Widgets da interface,
    # para efetivar as alterações neles.
    global button_create_svm
    global label_time_elapsed

    # Atribuir configurações para os Widgets da interface.
    button_create_svm.config(state=tkinter.constants.DISABLED)
    label_time_elapsed.config(text=f'Tempo Gasto: {round(time.time() - time_elapsed, 2)}s')

    # Salvar o SVM.
    joblib.dump(svm, 'models/svm.pkl')

    # Obter o vetor de predições.
    svm_prediction = svm.predict(numpy.array(x_projection_testing))

    # Calcular a acurácia para os testes.
    svm_accuracy = sklearn.metrics.accuracy_score(y_testing, svm_prediction)

    # Calcular a Matriz de Confusão para os testes.
    svm_confusion_matrix = sklearn.metrics.confusion_matrix(y_testing, svm_prediction)

    # Plotar em um mapa de calor a Matriz de Confusão.
    matplotlib.pyplot.figure(figsize=(10, 10))

    seaborn.heatmap(svm_confusion_matrix, annot=True, annot_kws={'size': 16}, fmt='g')

    matplotlib.pyplot.title(f'Matriz de Confusão do SVM\nAcurácia: {round(svm_accuracy * 100)}%')

    global svm_figure

    # É necessário salvar a imagem como um PNG e reabrí-la com o PIL.Image,
    # para que ela possa aparecer na interface do Tkinter.
    svm_figure = matplotlib.pyplot.savefig('models/svm_confusion_matrix.png')
    matplotlib.pyplot.close(svm_figure)
    
    # Ler a Matriz de Confusão.
    svm_figure = PIL.Image.open('models/svm_confusion_matrix.png')

    # Diminuir o tamanho da Matriz de Confusão para que ela caiba na interface.
    # O limite de largura de 768 Pixels foi estipulado com base em resoluções comuns para Notebooks.
    while svm_figure.width >= 768:
        svm_figure = svm_figure.resize((round(svm_figure.width * 0.75), round(svm_figure.height * 0.75)))

    # Converter a Matriz de Confusão para um formato suportado pela interface do Tkinter.
    svm_figure = PIL.ImageTk.PhotoImage(svm_figure)

    # Declarar como global os Widgets da interface,
    # para efetivar as alterações neles.
    global label_image

    # Atribuir configurações para os Widgets da interface.
    label_image.config(image=svm_figure)

# Função para criar e salvar o MLP.
# Para o MLP, utiliza-se um Layer de Input de uma única dimensão com valor 64,
# já que todas as projeções terão esse tamanho.
# Logo em seguida, tem-se dois Layers Dense,
# ambos com 128 neurônios e ativação Relu.
# Por último, um Layer Dense para Output com exatamente 10 neurônios e ativação Softmax,
# uma vez que tem-se 10 resultados possíveis: os dígitos de 0 até 9.
def create_mlp() -> None:
    time_elapsed = time.time()

    mlp = tensorflow.keras.models.Sequential()
    mlp.add(tensorflow.keras.layers.Input(64))
    mlp.add(tensorflow.keras.layers.Dense(128, activation=tensorflow.nn.relu))
    mlp.add(tensorflow.keras.layers.Dense(128, activation=tensorflow.nn.relu))
    mlp.add(tensorflow.keras.layers.Dense(10, activation=tensorflow.nn.softmax))
    mlp.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    # Treinar o MLP.
    mlp.fit(numpy.array(x_projection_training), y_training, epochs=12)

    # Declarar como global os Widgets da interface,
    # para efetivar as alterações neles.
    global button_create_mlp
    global button_create_svm
    global label_time_elapsed

    # Atribuir configurações para os Widgets da interface.
    button_create_mlp.config(state=tkinter.constants.DISABLED)
    button_create_svm.config(state=tkinter.constants.ACTIVE)
    label_time_elapsed.config(text=f'Tempo Gasto: {round(time.time() - time_elapsed, 2)}s')

    # Salvar o MLP.
    mlp.save('models/mlp.model')

    # Obter o vetor de predições.
    # Para o caso do MLP, é necessário utilizar-se o método "argmax" do Numpy,
    # uma vez que que as predições estão contidas em vetores de probabilidade,
    # sendo a probabilidade mais alta a predição.
    mlp_prediction = numpy.argmax(mlp.predict(numpy.array(x_projection_testing)), axis=1)

    # Calcular a acurácia para os testes.
    mlp_accuracy = sklearn.metrics.accuracy_score(y_testing, mlp_prediction)

    # Calcular a matriz de confusão para os testes.
    mlp_confusion_matrix = sklearn.metrics.confusion_matrix(y_testing, mlp_prediction)

    # Plotar em um mapa de calor a matriz de confusão.
    matplotlib.pyplot.figure(figsize=(10, 10))

    seaborn.heatmap(mlp_confusion_matrix, annot=True, annot_kws={'size': 16}, fmt='g')

    matplotlib.pyplot.title(f'Matriz de Confusão do MLP\nAcurácia: {round(mlp_accuracy * 100)}%')

    global mlp_figure

    # É necessário salvar a imagem como um PNG e reabrí-la com o PIL.Image,
    # para que ela possa aparecer na interface do Tkinter.
    mlp_figure = matplotlib.pyplot.savefig('models/mlp_confusion_matrix.png')
    matplotlib.pyplot.close(mlp_figure)
    
    # Ler a Matriz de Confusão.
    mlp_figure = PIL.Image.open('models/mlp_confusion_matrix.png')

    # Diminuir o tamanho da Matriz de Confusão para que ela caiba na interface.
    # O limite de largura de 768 Pixels foi estipulado com base em resoluções comuns para Notebooks.
    while mlp_figure.width >= 768:
        mlp_figure = mlp_figure.resize((round(mlp_figure.width * 0.75), round(mlp_figure.height * 0.75)))

    # Converter a Matriz de Confusão para um formato suportado pela interface do Tkinter.
    mlp_figure = PIL.ImageTk.PhotoImage(mlp_figure)

    # Declarar como global os Widgets da interface,
    # para efetivar as alterações neles.
    global label_image

    # Atribuir configurações para os Widgets da interface.
    label_image.config(image=mlp_figure)

# Função para interpolar a projeção de uma imagem para 32 elementos.
# Com a escolha do número 32,
# almeja-se aumentar o número de características de cada imagem.
def interpolate_projection(projection: list) -> list:
    function = scipy.interpolate.interp1d(numpy.arange(0, len(projection)), projection)

    return function(numpy.linspace(0.0, len(projection) - 1, 32)).tolist()

# Função para calcular a projeção horizontal de uma imagem.
# A projeção horizontal se dá pela soma dos valores dos Pixels de todas as linhas.
def calculate_h_projection(image: numpy.ndarray, h: int, w: int) -> list:
    h_projection = []

    for i in range(h):
        pixel_value_total = 0

        for j in range(w):
            pixel_value_total += image[i][j]

        h_projection.append(pixel_value_total)

    return h_projection

# Função para calcular a projeção vertical de uma imagem.
# A projeção vertical se dá pela soma dos valores dos Pixels de todas as colunas.
def calculate_v_projection(image: numpy.ndarray, h: int, w: int) -> list:
    v_projection = []

    for i in range(h):
        pixel_value_total = 0

        for j in range(w):
            pixel_value_total += image[j][i]

        v_projection.append(pixel_value_total)

    return v_projection

# Função para calcular a projeção de uma imagem.
def calculate_projection(image: numpy.ndarray) -> list:
    # Obter a altura e a largura da imagem.
    h, w = image.shape

    h_projection = calculate_h_projection(image, h, w)
    v_projection = calculate_v_projection(image, h, w)

    h_projection = interpolate_projection(h_projection)
    v_projection = interpolate_projection(v_projection)

    # Concatenar as projeções horizontal e vertical.
    projection = h_projection + v_projection

    # Normalizar os valores da projeção para um intervalo de 0 até 1.
    return tensorflow.keras.utils.normalize(numpy.array(projection), axis=0)[0].tolist()

# Função para processar as imagens do MNIST,
# transformando-as em projeções.
def process_mnist() -> None:
    time_elapsed = time.time()

    global x_projection_testing
    global x_projection_training

    # Calcular as projeções das imagens dos Data Sets do MNIST.
    for image in x_testing:
        x_projection_testing.append(calculate_projection(image))
    for image in x_training:
        x_projection_training.append(calculate_projection(image))

    # Declarar como global os Widgets da interface,
    # para efetivar as alterações neles.
    global button_process_mnist
    global button_create_mlp
    global label_time_elapsed

    # Atribuir configurações para os Widgets da interface.
    button_process_mnist.config(state=tkinter.constants.DISABLED)
    button_create_mlp.config(state=tkinter.constants.ACTIVE)
    label_time_elapsed.config(text=f'Tempo Gasto: {round(time.time() - time_elapsed, 2)}s')

# Função para carregar os Data Sets do MNIST.
def load_mnist() -> None:
    time_elapsed = time.time()

    mnist = tensorflow.keras.datasets.mnist

    global x_testing
    global x_training
    global y_testing
    global y_training

    # Carregar os Data Sets de treino e testes do MNIST.
    (x_training, y_training), (x_testing, y_testing) = mnist.load_data()
    
    # Declarar como global os Widgets da interface,
    # para efetivar as alterações neles.
    global button_load_mnist
    global button_process_mnist
    global label_time_elapsed

    # Atribuir configurações para os Widgets da interface.
    button_load_mnist.config(state=tkinter.constants.DISABLED)
    button_process_mnist.config(state=tkinter.constants.ACTIVE)
    label_time_elapsed.config(text=f'Tempo Gasto: {round(time.time() - time_elapsed, 2)}s')

def main() -> None:
    # Declarar como global os Widgets da interface,
    # para efetivar as alterações neles.
    global canvas
    global button_load_mnist
    global button_process_mnist
    global button_create_mlp
    global button_create_svm
    global label_time_elapsed
    global label_image

    # Inicializar a interface.
    canvas.create_window((0, 0), anchor="nw", window=second_frame)

    # Atribuir configurações para os Widgets da interface.
    button_load_mnist.config(command=load_mnist, state=tkinter.constants.ACTIVE, text='Carregar MNIST')
    button_process_mnist.config(command=process_mnist, state=tkinter.constants.DISABLED, text='Processar MNIST')
    button_create_mlp.config(command=create_mlp, state=tkinter.constants.DISABLED, text='Criar MLP')
    button_create_svm.config(command=create_svm, state=tkinter.constants.DISABLED, text='Criar SVM')

    # Posicionar os Widgets na interface.
    button_load_mnist.grid(row=0, column=0, padx=12, pady=12)
    button_process_mnist.grid(row=0, column=1, padx=12, pady=12)
    button_create_mlp.grid(row=0, column=2, padx=12, pady=12)
    button_create_svm.grid(row=0, column=3, padx=12, pady=12)
    label_time_elapsed.grid(row=1, column=0, padx=12, pady=12)
    label_image.grid(row=2, columnspan=4, padx=12, pady=12)

    root.mainloop()

if __name__ == '__main__':
    main()
