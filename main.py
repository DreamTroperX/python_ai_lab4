import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# Parameters
INPUT_SIZE = 400  # 20x20 пікселів
HIDDEN_SIZE = 50  # Кількість нейронів у прихованому шарі
OUTPUT_SIZE = 5  # Кількість класів (2, 3, 4, 5, 7)
LEARNING_RATE = 0.1
EPOCHS = 2000

# Ініціалізація ваг та зсувів
np.random.seed(42)
weights_input_hidden = np.random.uniform(-1, 1, (INPUT_SIZE, HIDDEN_SIZE))
weights_hidden_output = np.random.uniform(-1, 1, (HIDDEN_SIZE, OUTPUT_SIZE))
bias_hidden = np.random.uniform(-1, 1, (HIDDEN_SIZE,))
bias_output = np.random.uniform(-1, 1, (OUTPUT_SIZE,))


def sigmoid(x):
    # Сигмоїдна активаційна функція
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    # Похідна сигмоїдної функції для градієнтного спуску
    return x * (1 - x)


def forward_propagation(inputs):
    # Прямий прохід через мережу
    # Вхідний шар -> Прихований шар
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Прихований шар -> Вихідний шар
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output, output_layer_output


def backward_propagation(inputs, hidden_output, actual_output, predicted_output):
    global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

    # Помилка на виході
    output_error = actual_output - predicted_output
    # Оновлення для вихідного шару
    output_delta = output_error * sigmoid_derivative(predicted_output)

    # Помилка для прихованого шару
    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    # Оновлення для прихованого шару
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # Оновлення ваг та зсувів
    weights_hidden_output += np.dot(hidden_output.T, output_delta) * LEARNING_RATE
    weights_input_hidden += np.dot(inputs.T, hidden_delta) * LEARNING_RATE
    bias_output += np.sum(output_delta, axis=0) * LEARNING_RATE
    bias_hidden += np.sum(hidden_delta, axis=0) * LEARNING_RATE


def preprocess_image(image_path):
    # Обробка зображення: переведення в градації сірого, зміна розміру, нормалізація
    image = Image.open(image_path).convert("L")  # Перетворення в градації сірого
    image = image.resize((20, 20))  # Зміна розміру до 20x20
    return np.array(image).flatten() / 255.0  # Нормалізація до діапазону [0, 1]


def train_network(training_data, labels):
    for epoch in range(EPOCHS):
        # Рандомізація порядку подання зображень
        indices = np.random.permutation(len(training_data)) # зображення подаються в нейронну мережу у випадковому порядку на кожній епосі
        for i in indices:
            inputs, label = training_data[i], labels[i]
            inputs = np.array(inputs).reshape(1, -1)
            label = np.array(label).reshape(1, -1)

            # Прямий та зворотний прохід
            hidden_output, predicted_output = forward_propagation(inputs)
            backward_propagation(inputs, hidden_output, label, predicted_output)

        # Виведення помилки кожні 100 епох
        if epoch % 100 == 0:
            loss = np.mean((labels - predicted_output) ** 2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")


def predict_outputs(test_image_path):
    # Прогнозування для тестового зображення
    test_input = preprocess_image(test_image_path)
    _, predicted_output = forward_propagation(test_input)
    return predicted_output


def visualize_results(test_image, similarities):
    # Візуалізація результатів: тестове зображення та схожість з класами
    fig, axes = plt.subplots(1, 6, figsize=(18, 4))

    # Тестове зображення
    test_img = Image.open(test_image)
    axes[0].imshow(test_img, cmap='gray')
    axes[0].set_title("Test Image")
    axes[0].axis("off")

    # Порівняння з кожним класом
    classes = ["2", "3", "4", "5", "7"]
    for i, similarity in enumerate(similarities):
        class_image_path = f"data/ideal/{classes[i]}.png"
        class_image = Image.open(class_image_path)
        axes[i + 1].imshow(class_image, cmap='gray')
        axes[i + 1].set_title(f"Class {classes[i]}\n{similarity * 100:.2f}%")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig("results.png")
    plt.show()


def main():
    # Завантаження навчальних даних
    data_dir = "data"
    classes = ["2", "3", "4", "5", "7"]
    training_data = []
    labels = []

    # Підготовка даних для кожного класу
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for filename in os.listdir(class_dir):
            training_data.append(preprocess_image(os.path.join(class_dir, filename)))
            label = np.zeros(len(classes))
            label[i] = 1
            labels.append(label)

    training_data = np.array(training_data)
    labels = np.array(labels)

    train_network(training_data, labels)

    # Тестування
    test_image_path = input("Enter path to test image: ") # data/test/test1.png
    predictions = predict_outputs(test_image_path)
    visualize_results(test_image_path, predictions.flatten())


if __name__ == "__main__":
    main()
