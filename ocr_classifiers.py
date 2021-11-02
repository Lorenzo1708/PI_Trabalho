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
import tkinter.constants
import tkinter.filedialog
import tensorflow
import matplotlib.pyplot


root = tkinter.Tk()
root.title('MLP e SVM')

button_load_mnist = tkinter.Button(root, text='')
button_create_mlp = tkinter.Button(root, text='')
button_create_svm = tkinter.Button(root, text='')
label_time_passed = tkinter.Label(root, text='')
label_accuracy = tkinter.Label(root, text='')
label_confusion_matrix = tkinter.Label(root, image='')


x_testing = []
x_training = []
y_testing = []
y_training = []
x_projection_testing = []
x_projection_training = []


# Função para plotar as predições realizadas pelo MLP para o Data Set de testes do MNIST.
def plot_mlp(mlp) -> None:
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

    matplotlib.pyplot.title('Matriz de Confusão do MLP')

    global mlp_figure

    mlp_figure = matplotlib.pyplot.savefig('output/mlp_confusion_matrix.png')

    matplotlib.pyplot.close(mlp_figure)

    mlp_figure = PIL.Image.open('output/mlp_confusion_matrix.png')

    while mlp_figure.height > 768 or mlp_figure.width > 768:
        mlp_figure = mlp_figure.resize((round(mlp_figure.height * 0.75), round(mlp_figure.width * 0.75)))

    mlp_figure = PIL.ImageTk.PhotoImage(mlp_figure)

    global label_accuracy
    global label_confusion_matrix

    label_accuracy.config(text=f'Acurácia: {round(mlp_accuracy * 100)}%')
    label_confusion_matrix.config(image=mlp_figure)


def create_mlp() -> None:
    time_passed = time.time()

    mlp = tensorflow.keras.models.Sequential()

    mlp.add(tensorflow.keras.layers.Input(64))

    mlp.add(tensorflow.keras.layers.Dense(128, activation=tensorflow.nn.relu))
    mlp.add(tensorflow.keras.layers.Dense(128, activation=tensorflow.nn.relu))

    mlp.add(tensorflow.keras.layers.Dense(10, activation=tensorflow.nn.softmax))

    mlp.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    mlp.fit(numpy.array(x_projection_training), y_training, epochs=12)

    global button_create_mlp
    global label_time_passed

    button_create_mlp.config(text='Criar o MLP', state=tkinter.constants.DISABLED)
    label_time_passed.config(text=f'Tempo Total: {round(time.time() - time_passed, 2)}s')

    mlp.save('output/mlp.model')

    plot_mlp(mlp)


# Função para plotar as predições realizadas pelo SVM para o Data Set de testes do MNIST.
def plot_svm(svm) -> None:
    # Obter o vetor de predições.
    svm_prediction = svm.predict(numpy.array(x_projection_testing))

    # Calcular a acurácia para os testes.
    svm_accuracy = sklearn.metrics.accuracy_score(y_testing, svm_prediction)

    # Calcular a matriz de confusão para os testes.
    svm_confusion_matrix = sklearn.metrics.confusion_matrix(y_testing, svm_prediction)

    # Plotar em um mapa de calor a matriz de confusão.
    matplotlib.pyplot.figure(figsize=(10, 10))

    seaborn.heatmap(svm_confusion_matrix, annot=True, annot_kws={'size': 16}, fmt='g')

    matplotlib.pyplot.title('Matriz de Confusão do SVM')

    global svm_figure

    svm_figure = matplotlib.pyplot.savefig('output/svm_confusion_matrix.png')

    matplotlib.pyplot.close(svm_figure)

    svm_figure = PIL.Image.open('output/svm_confusion_matrix.png')

    while svm_figure.height > 768 or svm_figure.width > 768:
        svm_figure = svm_figure.resize((round(svm_figure.height * 0.75), round(svm_figure.width * 0.75)))

    svm_figure = PIL.ImageTk.PhotoImage(svm_figure)

    global label_accuracy
    global label_confusion_matrix

    label_accuracy.config(text=f'Acurácia: {round(svm_accuracy * 100)}%')
    label_confusion_matrix.config(image=svm_figure)


def create_svm() -> None:
    time_passed = time.time()

    svm = sklearn.svm.SVC(C=12)

    svm.fit(numpy.array(x_projection_training), y_training)

    global button_create_svm
    global label_time_passed

    button_create_svm.config(text='Criar o SVM', state=tkinter.constants.DISABLED)
    label_time_passed.config(text=f'Tempo Total: {round(time.time() - time_passed, 2)}s')

    joblib.dump(svm, 'output/svm.pkl')
    
    plot_svm(svm)


def interpolate_projection(projection: list) -> list:
    function = scipy.interpolate.interp1d(numpy.arange(0, len(projection)), projection)

    return function(numpy.linspace(0.0, len(projection) - 1, 32)).tolist()


def calculate_h_projection(image: numpy.ndarray, h: int, w: int) -> list:
    h_projection = []

    for i in range(h):
        pixel_color_count = 0

        for j in range(w):
            pixel_color_count += image[i][j]

        h_projection.append(pixel_color_count)

    return h_projection


def calculate_v_projection(image: numpy.ndarray, h: int, w: int) -> list:
    v_projection = []

    for i in range(h):
        pixel_color_count = 0

        for j in range(w):
            pixel_color_count += image[j][i]

        v_projection.append(pixel_color_count)

    return v_projection


def calculate_projection(image: numpy.ndarray) -> list:
    h, w = image.shape

    h_projection = calculate_h_projection(image, h, w)

    v_projection = calculate_v_projection(image, h, w)

    h_projection = interpolate_projection(h_projection)

    v_projection = interpolate_projection(v_projection)

    projection = h_projection + v_projection

    return tensorflow.keras.utils.normalize(numpy.array(projection), axis=0)[0].tolist()


def load_mnist() -> None:
    time_passed = time.time()

    mnist = tensorflow.keras.datasets.mnist

    global x_testing
    global x_training
    global y_testing
    global y_training

    (x_training, y_training), (x_testing, y_testing) = mnist.load_data()

    global x_projection_testing
    global x_projection_training

    for image in x_testing:
        x_projection_testing.append(calculate_projection(image))

    for image in x_training:
        x_projection_training.append(calculate_projection(image))

    global button_load_mnist
    global button_create_mlp
    global button_create_svm
    global label_time_passed

    button_load_mnist.config(state=tkinter.constants.DISABLED)
    button_create_mlp.config(state=tkinter.constants.ACTIVE)
    button_create_svm.config(state=tkinter.constants.ACTIVE)
    label_time_passed.config(text=f'Tempo Total: {round(time.time() - time_passed, 2)}s')


def main() -> None:
    global button_load_mnist
    global button_create_mlp
    global button_create_svm
    global label_time_passed
    global label_accuracy
    global label_confusion_matrix

    button_load_mnist.config(text='Carregar MNIST', command=load_mnist, state=tkinter.constants.ACTIVE)
    button_create_mlp.config(text='Criar o MLP', command=create_mlp, state=tkinter.constants.DISABLED)
    button_create_svm.config(text='Criar o SVM', command=create_svm, state=tkinter.constants.DISABLED)

    button_load_mnist.grid(row=0, column=0, padx=12, pady=12)
    button_create_mlp.grid(row=0, column=1, padx=12, pady=12)
    button_create_svm.grid(row=0, column=2, padx=12, pady=12)
    label_time_passed.grid(row=1, columnspan=3, padx=12, pady=12)
    label_accuracy.grid(row=2, columnspan=3, padx=12, pady=12)
    label_confusion_matrix.grid(row=3, columnspan=3, padx=12, pady=12)

    root.mainloop()


if __name__ == '__main__':
    main()
