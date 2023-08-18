import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow.keras.datasets import mnist # Библиотека образцов рукописных цифр (В базе данных 60тыс изображений в обучающей выборке и 10тыс в тестовой)
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt


# Загрузка и предобработка данных
# x_train - изображения цифр обучающей выборки
# y_train - вектор соотвествующих значений цифр (например, если на i-m изображении нарисована 5, то y_train[i] = 5)
# x_test - изображения цифр тестовой выборки
# y_test - вектор соответсвующих значений цифр для тестовой выборки
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# 255 - максимальное значение поэтому числа будут вещественными от 0 до 1 (Сделали нормализацию)
x_train = x_train / 255
x_test = x_test / 255
# Представляем цифру в виде вектора
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Проверка наличия сохраненной модели
if os.path.exists("trained_model.h5"):
    model = keras.models.load_model("trained_model.h5")
    print("Загружена сохраненная модель.")
else:
    # Создание модели
    model = keras.Sequential([
        #Flatten - преобразовывает входную матрицу в слой состоящий из вектора длиной 784 элемента
        Flatten(input_shape=(28, 28, 1)),
        #Связываем вход с каждым нейронном (со всеми 128 скрытыми нейронами)
        #Dense - указываем что будет 128 скрытых нейронов и функция активации relu
        Dense(128, activation='relu'),
        #Dense - указываем что будет 10 выходных нейронов и функция активации softmax
        Dense(10, activation='softmax')
    ])
    # Процесс компиляции ИИ с оптимизацией по Adam и критерием - категориальная кросс-энтропия (Критерий качества - используется для классификации) - уменьшаем кол-во ошибок ИИ
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    epochs = int(input("Введите количество поколений для обучения: "))
    # Обучение модели
    # x_train - входное обучающее множество
    # y_train_cat - требуемое значение на выходе ИИ (виде вектора)
    # batch_size - размер батча (после каждого 32 изображения идёт корректировка весового коеффициента)
    #epochs - кол-во эпох для обучения ИИ
    #validation_split - разбиение обучающей выборки на обучающую и проверочную (20№ картинок берём для тестирования)
    model.fit(x_train, y_train_cat, batch_size=32, epochs=epochs, validation_split=0.2)
    print("Модель обучена.")

# Функция для распознавания изображения и вывода результата
def recognize_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = 1 - image_array  # Инвертируем цвета (белый фон, черная цифра)
    image_array = np.expand_dims(image_array, axis=0)
    result = model.predict(image_array)
    digit = np.argmax(result)
    confidence = np.max(result)
    is_digit_image = is_digit(image_array)
    return digit, confidence, image_array, is_digit_image

# Функция для сбора неверных предсказаний
def collect_mistakes(predictions, correct_labels, images):
    mistakes = []
    for i in range(len(predictions)):
        if predictions[i] != correct_labels[i]:
            mistakes.append(images[i])
    return np.array(mistakes)

# Функция для проверки, является ли изображение цифрой
def is_digit(image_array):
    # Применяем алгоритмы обработки изображений для определения, является ли изображение цифрой
    # Возвращаем True, если изображение является цифрой, и False в противном случае
    # Пример простой проверки с использованием порогового значения
    threshold = 0.2  # Пороговое значение для классификации
    mean_intensity = np.mean(image_array)
    
    if mean_intensity < threshold:
        return True
    else:
        return False

while True:
    # Предупреждение для выбора действия
    action = input("Для распознавания картинки введите 'y', для выхода введите 'n': ")
    if action.lower() == 'n':
        save_model = input("Хотите ли вы сохранить обученную модель? (y/n): ")
        if save_model.lower() == 'y':
            model.save("trained_model.h5")
            print("Модель сохранена.")
        break
    elif action.lower() == 'y':
        # Выбор файла с изображением
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if not image_path:
            continue

        digit, confidence, image_array, is_digit_image = recognize_image(image_path)
        
        if not is_digit_image:
            print("На изображении не цифра.")
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            continue

        print('Распознанная цифра:', digit)
        print('Уверенность:', confidence)
        # Отображение выбранной картинки
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        while True:
            try:
                correct_digit = int(input("Введите правильную цифру (от 0 до 9): "))
                if correct_digit < 0 or correct_digit > 9:
                    raise ValueError
                break
            except ValueError:
                print("Некорректный ввод. Пожалуйста, введите цифру от 0 до 9.")
        if digit != correct_digit:
            wrong_predictions = collect_mistakes([digit], [correct_digit], [image_array])
            wrong_predictions = np.reshape(wrong_predictions, (-1, 28, 28, 1))
            wrong_labels = keras.utils.to_categorical([correct_digit], 10)
            epochs = 10
            model.fit(wrong_predictions, wrong_labels, batch_size=32, epochs=epochs)
            print("Модель дообучена.")
        else:
            print("Правильное предсказание.")
    else:
        print("Некорректный ввод. Пожалуйста, введите 'y' или 'n'.")