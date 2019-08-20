import os
import pygame
import numpy as np
from tkinter import *
from tkinter import messagebox
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import pandas as pd

stdout = sys.__stdout__
stderr = sys.__stderr__
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')


class pixel(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (255, 255, 255)
        self.neighbors = []

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.x + self.width, self.y + self.height))

    def getNeighbors(self, g):

        j = self.x // 20
        i = self.y // 20
        rows = 28
        cols = 28

        if i < cols - 1:
            self.neighbors.append(g.pixels[i + 1][j])
        if i > 0:
            self.neighbors.append(g.pixels[i - 1][j])
        if j < rows - 1:
            self.neighbors.append(g.pixels[i][j + 1])
        if j > 0:
            self.neighbors.append(g.pixels[i][j - 1])

        if j > 0 and i > 0:
            self.neighbors.append(g.pixels[i - 1][j - 1])

        if j + 1 < rows and i > -1 and i - 1 > 0:
            self.neighbors.append(g.pixels[i - 1][j + 1])

        if j - 1 < rows and i < cols - 1 and j - 1 > 0:
            self.neighbors.append(g.pixels[i + 1][j - 1])

        if j < rows - 1 and i < cols - 1:
            self.neighbors.append(g.pixels[i + 1][j + 1])


class grid(object):
    pixels = []

    def __init__(self, row, col, width, height):
        self.rows = row
        self.cols = col
        self.len = row * col
        self.width = width
        self.height = height
        self.generate_pixels()
        pass

    def draw(self, surface):
        for row in self.pixels:
            for col in row:
                col.draw(surface)

    def generate_pixels(self):
        x_gap = self.width // self.cols
        y_gap = self.height // self.rows
        self.pixels = []
        for r in range(self.rows):
            self.pixels.append([])
            for c in range(self.cols):
                self.pixels[r].append(pixel(x_gap * c, y_gap * r, x_gap, y_gap))

        for r in range(self.rows):
            for c in range(self.cols):
                self.pixels[r][c].getNeighbors(self)

    def clicked(self, pos):
        try:
            t = pos[0]
            w = pos[1]
            g1 = int(t) // self.pixels[0][0].width
            g1 = int(t) // self.pixels[0][0].width
            g2 = int(w) // self.pixels[0][0].height

            return self.pixels[g2][g1]
        except:
            pass

    def convert_binary(self):
        li = self.pixels

        newMatrix = [[] for x in range(len(li))]

        for i in range(len(li)):
            for j in range(len(li[i])):
                if li[i][j].color == (255, 255, 255):
                    newMatrix[i].append(0)
                else:
                    newMatrix[i].append(1)

        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")

        Y_train = train["label"]
        X_train = train.drop(labels=["label"], axis=1)
        X_train = X_train / 255.0
        X_test = test / 255.0
        X_train = X_train.values.reshape(-1, 28, 28, 1)
        X_test = X_test.values.reshape(-1, 28, 28, 1)
        Y_train = to_categorical(Y_train, num_classes=10)

        for row in range(28):
            for x in range(28):
                X_test[0][row][x] = newMatrix[row][x]

        return X_test[:1]


def prediction_model(li):
    model = tf.keras.models.load_model('newmodel.h5')

    predictions = model.predict(li)
    print(predictions[0])
    t = (np.argmax(predictions[0]))
    print("I predict this number is a:", t)
    window = Tk()
    window.withdraw()
    messagebox.showinfo("Prediction", "I predict this number is a: " + str(t))
    window.destroy()


def main():
    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                li = g.convert_binary()
                prediction_model(li)
                g.generate_pixels()
            if pygame.mouse.get_pressed()[0]:

                pos = pygame.mouse.get_pos()
                clicked = g.clicked(pos)
                clicked.color = (0, 0, 0)
                for n in clicked.neighbors:
                    n.color = (0, 0, 0)

            if pygame.mouse.get_pressed()[2]:
                try:
                    pos = pygame.mouse.get_pos()
                    clicked = g.clicked(pos)
                    clicked.color = (255, 255, 255)
                except:
                    pass

        g.draw(win)
        pygame.display.update()


pygame.init()
width = height = 560
win = pygame.display.set_mode((width, height))
pygame.display.set_caption("Number Guesser")
g = grid(28, 28, width, height)
main()

pygame.quit()
quit()
