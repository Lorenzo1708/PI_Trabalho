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

root.title('OCR')

button_load_image = tkinter.Button(root, text='')
button_predict_mlp = tkinter.Button(root, text='')
button_predict_svm = tkinter.Button(root, text='')
label_time_passed = tkinter.Label(root, text='')
label_image = tkinter.Label(root, text='')
label_image_processed = tkinter.Label(root, text='')


digit_image = []

digit_projection = []


def predict_mlp() -> None:
    mlp = tensorflow.keras.models.load_model('output/mlp.model')

    mlp_prediction = numpy.argmax(mlp.predict(numpy.array(digit_projection)), axis=1)

    matplotlib.pyplot.figure()

    for index, digit in enumerate(digit_image):
        matplotlib.pyplot.subplot(math.ceil(len(mlp_prediction) / 4), 4, index + 1)

        matplotlib.pyplot.imshow(digit, cmap='gray')

        matplotlib.pyplot.title(mlp_prediction[index], fontdict={'color': '#F00', 'fontsize': 32})

    matplotlib.pyplot.suptitle('Predições do MLP')

    matplotlib.pyplot.tight_layout()

    global mlp_figure

    mlp_figure = matplotlib.pyplot.savefig('output/mlp_prediction.png')

    matplotlib.pyplot.close(mlp_figure)

    mlp_figure = PIL.ImageTk.PhotoImage(PIL.Image.open('output/mlp_prediction.png'))

    global label_image

    label_image.config(image=mlp_figure)


def predict_svm() -> None:
    svm = joblib.load('output/svm.pkl')

    svm_prediction = svm.predict(numpy.array(digit_projection))

    matplotlib.pyplot.figure()

    for index, digit in enumerate(digit_image):
        matplotlib.pyplot.subplot(math.ceil(len(svm_prediction) / 4), 4, index + 1)

        matplotlib.pyplot.imshow(digit, cmap='gray')

        matplotlib.pyplot.title(svm_prediction[index], fontdict={'color': '#F00', 'fontsize': 32})

    matplotlib.pyplot.suptitle('Predições do SVM')

    matplotlib.pyplot.tight_layout()

    global svm_figure

    svm_figure = matplotlib.pyplot.savefig('output/svm_prediction.png')

    matplotlib.pyplot.close(svm_figure)

    svm_figure = PIL.ImageTk.PhotoImage(PIL.Image.open('output/svm_prediction.png'))

    label_image = tkinter.Label(root, image=svm_figure)

    label_image.grid(row=2, columnspan=3, padx=12, pady=12)


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


def backGroundColorIsBlack(stats, image_processed, labels):
    backGroundIsBlack=False
    backGroundFound=False
    BACK_GROUND_LABEL=0
    x = stats[0] #menor posição no eixo X do digito
    y = stats[1] #menor posição no eixo Y do digito
    w = stats[2] #largura do digito
    h = stats[3] #altura do digito
    i=0
    #percorrer a imagem processada enquanto a i for menor que a altura e j for menor que a largura do digito
    #e o fundo não for encontrado
    while(i<h and backGroundFound==False):
        j=0
        while(j<w and backGroundFound==False):
            #se o pixel na posição atual pertence ao fundo o fundo foi encontrado    
            if labels[i+y][j+x] == BACK_GROUND_LABEL :
                backGroundFound=True
                #se o pixel na posição atual é preto
                if image_processed[i+y][j+x] == 0 :
                    backGroundIsBlack=True
            j+=1
        i+=1
    
    return backGroundIsBlack


def load_image() -> None:
    image_path = tkinter.filedialog.askopenfilename(filetypes=(('JPG', '*.jpg'), ('PNG', '*.png')), title='Carregar Imagem')

    time_passed = time.time()

    global image

    image = PIL.ImageTk.PhotoImage(PIL.Image.open(image_path))

    global image_processed

    image_processed = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image_processed = cv2.GaussianBlur(image_processed, (11, 11), 0)

    image_processed = cv2.threshold(image_processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    labels, stats_array = cv2.connectedComponentsWithStats(image_processed, 4)[1:3]

    if(backGroundColorIsBlack(stats_array[0], image_processed, labels)==False):
        image_processed = cv2.bitwise_not(image_processed)

        stats_array = cv2.connectedComponentsWithStats(image_processed, 4)[2]

    global digit_image

    global digit_projection

    for stats in stats_array[1:]:
        x = stats[0]
        y = stats[1]
        w = stats[2]
        h = stats[3]

        digit = []

        for i in range(h):
            digit.append([])

            digit[i] = [0 for j in range(w)]

        for i in range(h):
            for j in range(w):
                digit[i][j] = image_processed[i + y][j + x]

        border_value = round(h * 0.23)

        digit = cv2.copyMakeBorder(numpy.array(digit), border_value, border_value, border_value, border_value, cv2.BORDER_CONSTANT, 0)

        if h > 28:
            digit = cv2.resize(digit, (28, 28), cv2.INTER_AREA)
        else:
            digit = cv2.resize(digit, (28, 28), cv2.INTER_LINEAR)

        digit_image.append(digit)

        digit_projection.append(calculate_projection(digit))

        cv2.rectangle(image_processed, (x, y), (x + w, y + h), (255, 255, 255), 2)

    image_processed = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(image_processed))

    global label_image

    button_predict_mlp = tkinter.Button(root, text='Predizer com o MLP', command=predict_mlp)
    button_predict_svm = tkinter.Button(root, text='Predizer com o SVM', command=predict_svm)
    label_time_passed = tkinter.Label(root, text=f'Tempo Total: {round(time.time() - time_passed, 2)}s')
    label_image = tkinter.Label(root, image=image)
    label_image_processed = tkinter.Label(root, image=image_processed)

    button_predict_mlp.grid(row=0, column=1, padx=12, pady=12)
    button_predict_svm.grid(row=0, column=2, padx=12, pady=12)
    label_time_passed.grid(row=1, column=0, padx=12, pady=12)
    label_image.grid(row=2, columnspan=3, padx=12, pady=12)
    label_image_processed.grid(row=3, columnspan=3, padx=12, pady=12)


def main() -> None:
    button_load_image = tkinter.Button(root, text='Carregar Imagem', command=load_image)
    button_predict_mlp = tkinter.Button(root, text='Predizer com o MLP', state=tkinter.constants.DISABLED)
    button_predict_svm = tkinter.Button(root, text='Predizer com o SVM', state=tkinter.constants.DISABLED)

    button_load_image.grid(row=0, column=0, padx=12, pady=12)
    button_predict_mlp.grid(row=0, column=1, padx=12, pady=12)
    button_predict_svm.grid(row=0, column=2, padx=12, pady=12)

    root.mainloop()


if __name__ == '__main__':
    main()
