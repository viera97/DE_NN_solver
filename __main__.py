#!/home/viera/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
#=====================================#
# @author: Dayron Viera Quintero      #
#=====================================#
# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2

import os
import shutil

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.misc import derivative

from sympy import symbols, lambdify
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
import numpy as np

import matplotlib.pyplot as plt

import net

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox

def initial():

    if not 'img' in os.listdir():
        os.mkdir('img')

    for fig in os.listdir('img/'):
        if fig != 'default.png' and fig != 'default-window-example.png' and fig != 'window-example.png':
            os.remove(os.path.join('img/',fig))
    return

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(591, 417)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 171, 301))
        self.groupBox.setObjectName("groupBox")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 30, 151, 231))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.equation = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.equation.setLocale(QtCore.QLocale(QtCore.QLocale.Spanish, QtCore.QLocale.UnitedStates))
        self.equation.setObjectName("equation")
        self.verticalLayout_2.addWidget(self.equation)
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.initial_condition = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.initial_condition.setLocale(QtCore.QLocale(QtCore.QLocale.Spanish, QtCore.QLocale.UnitedStates))
        self.initial_condition.setObjectName("initial_condition")
        self.verticalLayout_2.addWidget(self.initial_condition)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.min = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.min.setObjectName("min")
        self.horizontalLayout.addWidget(self.min)
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        self.max = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.max.setObjectName("max")
        self.horizontalLayout.addWidget(self.max)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.label_8 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_8.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_2.addWidget(self.label_8)
        self.evaluation = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.evaluation.setObjectName("evaluation")
        self.verticalLayout_2.addWidget(self.evaluation)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_9 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_4.addWidget(self.label_9)
        self.neuronas = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.neuronas.setObjectName("neuronas")
        self.verticalLayout_4.addWidget(self.neuronas)
        self.horizontalLayout_3.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_12 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_5.addWidget(self.label_12)
        self.iteraciones = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.iteraciones.setObjectName("iteraciones")
        self.verticalLayout_5.addWidget(self.iteraciones)
        self.horizontalLayout_3.addLayout(self.verticalLayout_5)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_3)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.Calcular = QtWidgets.QPushButton(self.groupBox)
        self.Calcular.setGeometry(QtCore.QRect(10, 270, 61, 23))
        self.Calcular.setObjectName("Calcular")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(190, 10, 391, 381))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setGeometry(QtCore.QRect(-2, 24, 371, 311))
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.Picture = QtWidgets.QLabel(self.groupBox_2)
        self.Picture.setGeometry(QtCore.QRect(10, 30, 371, 311))
        self.Picture.setText("")

        try:
            self.Picture.setPixmap(QtGui.QPixmap("img/default.png"))
        except:
            pass

        self.Picture.setScaledContents(True)
        self.Picture.setObjectName("Picture")
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        self.label_7.setGeometry(QtCore.QRect(10, 350, 101, 21))
        self.label_7.setObjectName("label_7")
        self.Resultado = QtWidgets.QLabel(self.groupBox_2)
        self.Resultado.setGeometry(QtCore.QRect(90, 350, 101, 21))
        self.Resultado.setAcceptDrops(False)
        self.Resultado.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Resultado.setAutoFillBackground(False)
        self.Resultado.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Resultado.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Resultado.setScaledContents(False)
        self.Resultado.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Resultado.setObjectName("Resultado")
        self.savegraph = QtWidgets.QPushButton(self.groupBox_2)
        self.savegraph.setGeometry(QtCore.QRect(270, 350, 111, 23))
        self.savegraph.setCheckable(False)
        self.savegraph.setObjectName("savegraph")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionNew = QtWidgets.QAction(MainWindow)
        self.actionNew.setObjectName("actionNew")
        self.actionLoad = QtWidgets.QAction(MainWindow)
        self.actionLoad.setObjectName("actionLoad")
        self.actionAdvanced = QtWidgets.QAction(MainWindow)
        self.actionAdvanced.setObjectName("actionAdvanced")
        self.actionSalir = QtWidgets.QAction(MainWindow)
        self.actionSalir.setObjectName("actionSalir")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        global _translate

        _translate = QtCore.QCoreApplication.translate

        MainWindow.setWindowTitle(_translate("MainWindow", "PDE Solver"))
        self.groupBox.setTitle(_translate("MainWindow", "Problema"))
        self.label.setText(_translate("MainWindow", "Ecuación:"))
        self.equation.setToolTip(_translate("MainWindow", "<html><head/><body><p>Inserte una ecuación de la forma: <span style=\" color:#5500ff;\">dy/dx=f(x)</span><span style=\" color:#000000;\">.</span></p><p><span style=\" color:#ff0000;\">(Campo Necesario)</span></p></body></html>"))
        self.equation.setStatusTip(_translate("MainWindow", "Ecuación"))
        self.equation.setWhatsThis(_translate("MainWindow", "Insertar ecuación"))
        self.label_2.setText(_translate("MainWindow", "Condición Inicial:"))
        self.initial_condition.setToolTip(_translate("MainWindow", "<html><head/><body><p>Inserte una condición inicial <span style=\" color:#6f6f6f;\">(y(x</span><span style=\" color:#6f6f6f; vertical-align:sub;\">0</span><span style=\" color:#6f6f6f;\">)=y</span><span style=\" color:#6f6f6f; vertical-align:sub;\">0</span><span style=\" color:#6f6f6f;\">) </span>de la forma: <span style=\" color:#5500ff;\">x</span><span style=\" color:#5500ff; vertical-align:sub;\">0</span><span style=\" color:#5500ff;\">,y</span><span style=\" color:#5500ff; vertical-align:sub;\">0</span><span style=\" color:#000000;\">.</span></p><p><span style=\" color:#ff0000;\">(Campo opcional, Default: </span><span style=\" color:#5500ff;\">0,0</span><span style=\" color:#ff0000;\">)</span></p></body></html>"))
        self.initial_condition.setStatusTip(_translate("MainWindow", "Condición inicial"))
        self.initial_condition.setWhatsThis(_translate("MainWindow", "Condición inicial"))
        self.label_3.setText(_translate("MainWindow", "Mín:"))
        self.min.setToolTip(_translate("MainWindow", "<html><head/><body><p>Inserte mínimo del intervalo a graficar y calcular la red.</p><p><span style=\" color:#ff0000;\">(Campo opcional, Default: </span><span style=\" color:#5500ff;\">-10</span><span style=\" color:#ff0000;\">)</span></p></body></html>"))
        self.min.setStatusTip(_translate("MainWindow", "Mínimo del Intervalo"))
        self.min.setWhatsThis(_translate("MainWindow", "Mínimo del Intervalo"))
        self.label_4.setText(_translate("MainWindow", "Máx:"))
        self.max.setToolTip(_translate("MainWindow", "<html><head/><body><p>Inserte máximo del intervalo a graficar y calcular la red.</p><p><span style=\" color:#ff0000;\">(Campo opcional, Default: </span><span style=\" color:#5500ff;\">10</span><span style=\" color:#ff0000;\">)</span></p></body></html>"))
        self.max.setStatusTip(_translate("MainWindow", "Máximo del intervalo"))
        self.max.setWhatsThis(_translate("MainWindow", "Máximo del intervalo"))
        self.label_8.setText(_translate("MainWindow", "Evaluación:"))
        self.evaluation.setToolTip(_translate("MainWindow", "<html><head/><body><p>Evaluación de la función <span style=\" color:#5500ff;\">y</span> a partir de la aproximación de la red.</p><p><span style=\" color:#ff0000;\">(Campo opcional, Default: </span><span style=\" color:#5500ff;\">0</span><span style=\" color:#ff0000;\">)</span></p></body></html>"))
        self.evaluation.setStatusTip(_translate("MainWindow", "Evaluación"))
        self.evaluation.setWhatsThis(_translate("MainWindow", "Evaluación de la red luego de aproximar la solución de la ecuación diferencial."))
        self.label_9.setText(_translate("MainWindow", "Neuronas:"))
        self.neuronas.setToolTip(_translate("MainWindow", "<html><head/><body><p>Cantidad de neuronas en la capa interna.</p><p><span style=\" color:#ff0000;\">(Campo opcional, Default: </span><span style=\" color:#0000ff;\">100</span><span style=\" color:#ff0000;\">)</span></p></body></html>"))
        self.neuronas.setStatusTip(_translate("MainWindow", "Neuronas"))
        self.neuronas.setWhatsThis(_translate("MainWindow", "Cantidad de neuronas en la capa interna"))
        self.label_12.setText(_translate("MainWindow", "Iteraciones:"))
        self.iteraciones.setToolTip(_translate("MainWindow", "<html><head/><body><p>Cantidad de iteraciones para aproximar la red.</p><p><span style=\" color:#ff0000;\">(Campo opcional, Default: </span><span style=\" color:#5500ff;\">Hasta que el error de aproximación sea menor de 1e-4</span><span style=\" color:#ff0000;\">)</span></p></body></html>"))
        self.iteraciones.setStatusTip(_translate("MainWindow", "Iteraciones"))
        self.iteraciones.setWhatsThis(_translate("MainWindow", "Cantidad de iteraciones para calcular la red"))
        self.Calcular.setStatusTip(_translate("MainWindow", "Calcular"))
        self.Calcular.setText(_translate("MainWindow", "Calcular"))
        self.Calcular.setShortcut(_translate("MainWindow", "Return"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Resultado"))
        self.Picture.setStatusTip(_translate("MainWindow", "Gráfica de la red"))
        self.label_7.setText(_translate("MainWindow", "Evaluación:"))
        self.Resultado.setStatusTip(_translate("MainWindow", "Resultado de la evaluación"))
        self.Resultado.setText(_translate("MainWindow", "0"))
        self.savegraph.setToolTip(_translate("MainWindow", "Guarda los Gráfico"))
        self.savegraph.setStatusTip(_translate("MainWindow", "Guardar"))
        self.savegraph.setWhatsThis(_translate("MainWindow", "Guardar Gráfico"))
        self.savegraph.setText(_translate("MainWindow", "Guardar Gráfico"))
        self.savegraph.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionNew.setText(_translate("MainWindow", "Cargar Datos"))
        self.actionLoad.setText(_translate("MainWindow", "Salir"))
        self.actionAdvanced.setText(_translate("MainWindow", "Advanced"))
        self.actionSalir.setText(_translate("MainWindow", "Salir"))
        self.Calcular.clicked.connect(self.calc)
        self.savegraph.clicked.connect(self.save)

    def save(self):
        if len(os.listdir('img/'))==1:
            self.show_popup('No existe imagen')
            return
        else:
            dir = QFileDialog.getExistingDirectory()
            shutil.copyfile(f"img/fig{len(os.listdir('img/'))-1}.png",os.path.join(dir,f"fig{len(os.listdir('img/'))-1}.png"))
        return

    def calc(self):
        equation = self.equation.text()
        if not equation == '':
            equation = equation.split(' ')
            aux = ''
            for i in equation:
                aux += i
            equation = aux
            equation = equation.split('=')
            if len(equation) != 2 or (len(equation[0])==0 or len(equation[1])==0):
                self.show_popup('Sintaxis de ecuación incorrecta')
                return
            equation[0] = equation[0].split('/')
            if len(equation[0])!=2:
                self.show_popup('Sintaxis de ecuación incorrecta')
                return
            elif len(equation[0][0])==0 or len(equation[0][1])==0:
                self.show_popup('Sintaxis de ecuación incorrecta')
                return
            for i in range(len(equation[0])):
                equation[0][i] = equation[0][i].split('d')[1]
            initial_condition = self.initial_condition.text()
            if not initial_condition == '':
                initial_condition = initial_condition.split(',')
                if len(initial_condition) != 2:
                    self.show_popup('Sintaxis de condición inicial incorrecta')
                    return
                elif len(initial_condition[0]) == 0 or len(initial_condition[1]) == 0:
                    self.show_popup('Sintaxis de condición inicial incorrecta')
                    return
                try:
                    for i in range(len(initial_condition)):
                        initial_condition[i] = float(initial_condition[i])
                except:
                    self.show_popup('Sintaxis de condición inicial incorrecta')
                    return
            else:
                initial_condition = [0,0]
            evaluation = self.evaluation.text()
            if not evaluation == '':
                try:
                    evaluation = float(evaluation)
                except:
                    self.show_popup('Sintaxis de evaluación incorrecta')
                    return
            else:
                evaluation = 0

            min = self.min.text()
            max = self.max.text()

            if len(min) == 0 and len(max) == 0:
                min = -10
                max = 10
            elif len(min) != 0 and len(max) !=0:
                try:
                    min = float(min)
                    max = float(max)
                    if not min < max:
                        self.show_popup('Intervalo imposible')
                        return
                except:
                    self.show_popup('Sintaxis de intervalo incorrecta')
                    return
            elif len(min) != 0:
                try:
                    min = float(min)
                except:
                    self.show_popup('Sintaxis de intervalo incorrecta')
                    return
                max = 2*min
            else:
                try:
                    max = float(max)
                except:
                    self.show_popup('Sintaxis de intervalo incorrecta')
                    return
                min = max/2

            neuronas = self.neuronas.text()
            if neuronas == '':
                neuronas = 100
            else:
                try:
                    neuronas = int(neuronas)
                    if neuronas == 0:
                        self.show_popup('Número de neuronas no válido')
                        return
                except:
                    self.show_popup('Número de neuronas no válido')
                    return
            iteraciones = self.iteraciones.text()
            if iteraciones == '':
                iteraciones = 5000
            else:
                try:
                    iteraciones = int(iteraciones)
                    if iteraciones == 0:
                        self.show_popup('Número de iteraciones no válido')
                        return
                except:
                    self.show_popup('Número de iteraciones no válido')
                    return

            result = net.do_it(equation,initial_condition,evaluation,min,max,neuronas,iteraciones)

            if result == 1:
                self.show_popup('Sintaxis de ecuación incorrecta')
                return

            self.Picture.setPixmap(QtGui.QPixmap(f"img/fig{len(os.listdir('img/'))-1}.png"))
            self.Resultado.setText(_translate("MainWindow", str(result)))

        else:
            self.show_popup('Debe Insertar una ecuación')
            return

    def show_popup(self,text):
        msg = QMessageBox()
        msg.setWindowTitle('Error')
        msg.setText(text)

        x = msg.exec_()


if __name__ == "__main__":
    import sys
    initial()
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
