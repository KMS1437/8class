import numpy as np
import json

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.w7 = np.random.normal()
        self.w8 = np.random.normal()
        self.w9 = np.random.normal()
        self.w10 = np.random.normal()
        self.w11 = np.random.normal()
        self.w12 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.b4 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
        h2 = sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2)
        h3 = sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3)
        o1 = sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 2000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
                h2 = sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2)
                h3 = sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3)
                o1 = sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4)
                y_pred = o1
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Нейрон o1
                d_ypred_d_w10 = h1 * deriv_sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4)
                d_ypred_d_w11 = h2 * deriv_sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4)
                d_ypred_d_w12 = h3 * deriv_sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4)
                d_ypred_d_b4 = deriv_sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4)

                d_ypred_d_h1 = self.w10 * deriv_sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4)
                d_ypred_d_h2 = self.w11 * deriv_sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4)
                d_ypred_d_h3 = self.w12 * deriv_sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4)

                # Нейрон h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
                d_h1_d_w3 = x[2] * deriv_sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
                d_h1_d_b1 = deriv_sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)

                # Нейрон h2
                d_h2_d_w4 = x[0] * deriv_sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2)
                d_h2_d_w5 = x[1] * deriv_sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2)
                d_h2_d_w6 = x[2] * deriv_sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2)
                d_h2_d_b2 = deriv_sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2)

                # Нейрон h3
                d_h3_d_w7 = x[0] * deriv_sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3)
                d_h3_d_w8 = x[1] * deriv_sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3)
                d_h3_d_w9 = x[2] * deriv_sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3)
                d_h3_d_b3 = deriv_sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3)

                # Нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w3
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Нейрон h2
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w6
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Нейрон h3
                self.w7 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w7
                self.w8 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w8
                self.w9 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w9
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_b3

                # Нейрон o1
                self.w10 -= learn_rate * d_L_d_ypred * d_ypred_d_w10
                self.w11 -= learn_rate * d_L_d_ypred * d_ypred_d_w11
                self.w12 -= learn_rate * d_L_d_ypred * d_ypred_d_w12
                self.b4 -= learn_rate * d_L_d_ypred * d_ypred_d_b4

            # --- Подсчитываем общую потерю в конце каждой фазы
            if epoch % 100 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

with open("file.json", "r") as file:
    iris_data = json.load(file)

data = []
all_y_trues = []
for entry in iris_data:
    features = [float(entry["sepal_length"]), float(entry["sepal_width"]), float(entry["petal_length"])]
    data.append(features)
    if entry["species"] == "setosa":
        all_y_trues.append(0)
    elif entry["species"] == "versicolor":
        all_y_trues.append(1)
    elif entry["species"] == "virginica":
        all_y_trues.append(2)

data = np.array(data)
all_y_trues = np.array(all_y_trues)

network = OurNeuralNetwork()
network.train(data, all_y_trues)

example = np.array([6.3, 3.3, 6.0])
print(f"Цветок: {network.feedforward(example):.3f}")
