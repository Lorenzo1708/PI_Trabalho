"""
Discplina:
Processamento de Imagens (6498100)

Grupo:
Guilherme Schuffner Ribeiro - Matrícula: 655104
Henrique Dornas Mendes      - Matrícula: 651252
Lorenzo Duarte Costa        - Matrícula: 652641

Este Script deve ser executado só depois de se executar o arquivo "models.py",
uma vez que ele precisa carregar os arquivos dos modelos.

Para executá-lo, digite o seguinte comando no terminal:
python3 ocr.py
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
root.title('OCR')
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
button_load_image = tkinter.Button(second_frame)
button_process_image = tkinter.Button(second_frame)
button_predict_image_mlp = tkinter.Button(second_frame)
button_predict_image_svm = tkinter.Button(second_frame)
label_time_elapsed = tkinter.Label(second_frame)
label_image = tkinter.Label(second_frame)

image_path = ''

digit_image = []
digit_projection = []

# Função para predizer os dígitos da imagem usando o SVM.
def predict_image_svm() -> None:
    time_elapsed = time.time()

    # Carregar o SVM.
    svm = joblib.load('models/svm.pkl')

    # Obter o vetor de predições.
    svm_prediction = svm.predict(numpy.array(digit_projection))

    # Declarar como global os Widgets da interface,
    # para efetivar as alterações neles.
    global button_predict_image_svm
    global label_time_elapsed

    # Atribuir configurações para os Widgets da interface.
    button_predict_image_svm.config(state=tkinter.constants.DISABLED)
    label_time_elapsed.config(text=f'Tempo Gasto: {round(time.time() - time_elapsed, 2)}s')

    # Criar um Plot para os dígitos.
    matplotlib.pyplot.figure()

    for index, digit in enumerate(digit_image):
        matplotlib.pyplot.subplot(math.ceil(len(svm_prediction) / 4), 4, index + 1)

        matplotlib.pyplot.imshow(digit, cmap='gray')

        matplotlib.pyplot.title(svm_prediction[index], fontdict={'color': '#F00', 'fontsize': 32})

    matplotlib.pyplot.suptitle('Predições do SVM')

    matplotlib.pyplot.tight_layout()

    global svm_figure

    # É necessário salvar a imagem como um PNG e reabrí-la com o PIL.Image,
    # para que ela possa aparecer na interface do Tkinter.
    svm_figure = matplotlib.pyplot.savefig('models/svm_prediction.png')
    matplotlib.pyplot.close(svm_figure)

    # Ler a imagem.
    svm_figure = PIL.Image.open('models/svm_prediction.png')

    # Diminuir o tamanho da imagem para que ela caiba na interface.
    # O limite de largura de 768 Pixels foi estipulado com base em resoluções comuns para Notebooks.
    while svm_figure.width >= 768:
        svm_figure = svm_figure.resize((round(svm_figure.width * 0.75), round(svm_figure.height * 0.75)))

    # Converter a imagem para um formato suportado pela interface do Tkinter.
    svm_figure = PIL.ImageTk.PhotoImage(svm_figure)

    # Declarar como global os Widgets da interface,
    # para efetivar as alterações neles.
    global label_image

    # Atribuir configurações para os Widgets da interface.
    label_image.config(image=svm_figure)

# Função para predizer os dígitos da imagem usando o MLP.
def predict_image_mlp() -> None:
    time_elapsed = time.time()

    # Carregar o MLP.
    mlp = tensorflow.keras.models.load_model('models/mlp.model')

    # Obter o vetor de predições.
    # Para o caso do MLP, é necessário utilizar-se o método "argmax" do Numpy,
    # uma vez que que as predições estão contidas em vetores de probabilidade,
    # sendo a probabilidade mais alta a predição.
    mlp_prediction = numpy.argmax(mlp.predict(numpy.array(digit_projection)), axis=1)

    # Declarar como global os Widgets da interface,
    # para efetivar as alterações neles.
    global button_predict_image_mlp
    global label_time_elapsed

    # Atribuir configurações para os Widgets da interface.
    button_predict_image_mlp.config(state=tkinter.constants.DISABLED)
    label_time_elapsed.config(text=f'Tempo Gasto: {round(time.time() - time_elapsed, 2)}s')

    # Criar um Plot para os dígitos.
    matplotlib.pyplot.figure()

    for index, digit in enumerate(digit_image):
        matplotlib.pyplot.subplot(math.ceil(len(mlp_prediction) / 4), 4, index + 1)

        matplotlib.pyplot.imshow(digit, cmap='gray')

        matplotlib.pyplot.title(mlp_prediction[index], fontdict={'color': '#F00', 'fontsize': 32})

    matplotlib.pyplot.suptitle('Predições do MLP')

    matplotlib.pyplot.tight_layout()

    global mlp_figure

    # É necessário salvar a imagem como um PNG e reabrí-la com o PIL.Image,
    # para que ela possa aparecer na interface do Tkinter.
    mlp_figure = matplotlib.pyplot.savefig('models/mlp_prediction.png')
    matplotlib.pyplot.close(mlp_figure)

    # Ler a imagem.
    mlp_figure = PIL.Image.open('models/mlp_prediction.png')

    # Diminuir o tamanho da imagem para que ela caiba na interface.
    # O limite de largura de 768 Pixels foi estipulado com base em resoluções comuns para Notebooks.
    while mlp_figure.width >= 768:
        mlp_figure = mlp_figure.resize((round(mlp_figure.width * 0.75), round(mlp_figure.height * 0.75)))

    # Converter a imagem para um formato suportado pela interface do Tkinter.
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

# Função para processar uma imagem,
# segmentando seus dígitos e transformando-os em projeções.
def process_image() -> None:
    time_elapsed = time.time()

    global image

    # Leitura da imagem em tons de cinza
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Redução de ruídos da imagem
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Limiarizar a imagem para preto e branco.
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Obter a altura e a largura da imagem.
    h, w = image.shape

    # Se a imagem contem mais Pixels brancos do que pretos,
    # é porque o seu fundo é branco.
    # Para separar os digitos da imagem,
    # o fundo deve ser preto.
    # Por isso, é preciso inverter as cores da imagem.
    if cv2.countNonZero(image) >= (h * w) / 2:
        image = cv2.bitwise_not(image)

    global digit_image
    global digit_projection

    digit_image = []
    digit_projection = []

    # Segmentar os dígitos da imagem.
    stats = cv2.connectedComponentsWithStats(image, 4)[2][1:]

    image_copy = image.copy()

    # Armazenar as imagens e projeções dos dígitos.
    for index in range(len(stats)):
        x = stats[index][0] # Menor posição do dígito no eixo X.
        y = stats[index][1] # Menor posição do dígito no eixo Y.
        w = stats[index][2] # Largura do dígito.
        h = stats[index][3] # Altura do dígito.

        digit = []

        # Inicializar o Array do dígito como um fundo preto.
        for i in range(h):
            digit.append([])

            digit[i] = [0 for j in range(w)]

        # Copiar o dígito da imagem para o Array.
        for i in range(h):
            for j in range(w):
                digit[i][j] = image[i + y][j + x]

        # Definir que o valor da borda será o equivalente a 23% da altura do dígito.	
        # Esse valor foi estipulado baseando-se nos padrões de borda da base do MNIST.
        border_value = round(h * 0.23)

        # Acrescentar as bordas no dígito.
        digit = cv2.copyMakeBorder(numpy.array(digit), border_value, border_value, border_value, border_value, cv2.BORDER_CONSTANT, 0)

        # Redimensionar a imagem para assemelhá-la às imagens do MNIST.
        # Caso ela seja maior, utiliza-se uma interpolação de área,
        # caso contrário, utiliza-se uma interpolação linear.
        if h > 28:
            digit = cv2.resize(digit, (28, 28), cv2.INTER_AREA)
        else:
            digit = cv2.resize(digit, (28, 28), cv2.INTER_LINEAR)

        digit_image.append(digit)

        digit_projection.append(calculate_projection(digit))

        # Adicionar um retângulo na imagem,
        # destacando o dígito segmentado.
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 255, 255), 4)

    # Converter a imagem de um Numpy Array para PNG.
    image = PIL.Image.fromarray(image_copy)

    # Diminuir o tamanho da imagem para que ela caiba na interface.
    # O limite de largura de 768 Pixels foi estipulado com base em resoluções comuns para Notebooks.
    while image.width >= 768:
        image = image.resize((round(image.width * 0.75), round(image.height * 0.75)))

    # Converter a imagem para um formato suportado pela interface do Tkinter.
    image = PIL.ImageTk.PhotoImage(image)
    
    # Declarar como global os Widgets da interface,
    # para efetivar as alterações neles.
    global button_process_image
    global button_predict_image_mlp
    global button_predict_image_svmg
    global label_time_elapsed
    global label_image

    # Atribuir configurações para os Widgets da interface.
    button_process_image.config(state=tkinter.constants.DISABLED)
    button_predict_image_mlp.config(state=tkinter.constants.ACTIVE)
    button_predict_image_svm.config(state=tkinter.constants.ACTIVE)
    label_time_elapsed.config(text=f'Tempo Gasto: {round(time.time() - time_elapsed, 2)}s')
    label_image.config(image=image)

# Função para carregar uma imagem.
def load_image() -> None:
    global image_path
    
    # Abrir o Explorador de Arquivos e solicitar uma imagem no formato JPG ou PNG.
    image_path = tkinter.filedialog.askopenfilename(filetypes=(('JPG', '*.jpg'), ('PNG', '*.png')), title='Carregar Imagem')

    global image

    # Ler a imagem.
    image = PIL.Image.open(image_path)

    # Diminuir o tamanho da imagem para que ela caiba na interface.
    # O limite de largura de 768 Pixels foi estipulado com base em resoluções comuns para Notebooks.
    while image.width >= 768:
        image = image.resize((round(image.width * 0.75), round(image.height * 0.75)))

    # Converter a imagem para um formato suportado pela interface do Tkinter.
    image = PIL.ImageTk.PhotoImage(image)
    
    # Declarar como global os Widgets da interface,
    # para efetivar as alterações neles.
    global button_process_image
    global button_predict_image_mlp
    global button_predict_image_svm
    global label_image

    # Atribuir configurações para os Widgets da interface.
    button_process_image.config(state=tkinter.constants.ACTIVE)
    button_predict_image_mlp.config(state=tkinter.constants.DISABLED)
    button_predict_image_svm.config(state=tkinter.constants.DISABLED)
    label_image.config(image=image)

def main() -> None:
    # Declarar como global os Widgets da interface,
    # para efetivar as alterações neles.
    global canvas
    global button_load_image
    global button_process_image
    global button_predict_image_mlp
    global button_predict_image_svm
    global label_time_elapsed
    global label_image

    # Inicializar a interface.
    canvas.create_window((0, 0), anchor="nw", window=second_frame)

    # Atribuir configurações para os Widgets da interface.
    button_load_image.config(command=load_image, state=tkinter.constants.ACTIVE, text='Carregar Imagem')
    button_process_image.config(command=process_image, state=tkinter.constants.DISABLED, text='Processar Imagem')
    button_predict_image_mlp.config(command=predict_image_mlp, state=tkinter.constants.DISABLED, text='Predizer com MLP')
    button_predict_image_svm.config(command=predict_image_svm, state=tkinter.constants.DISABLED, text='Predizer com SVM')

    # Posicionar os Widgets na interface.
    button_load_image.grid(row=0, column=0, padx=12, pady=12)
    button_process_image.grid(row=0, column=1, padx=12, pady=12)
    button_predict_image_mlp.grid(row=0, column=2, padx=12, pady=12)
    button_predict_image_svm.grid(row=0, column=3, padx=12, pady=12)
    label_time_elapsed.grid(row=1, column=0, padx=12, pady=12)
    label_image.grid(row=2, columnspan=4, padx=12, pady=12)

    root.mainloop()

if __name__ == '__main__':
    main()
