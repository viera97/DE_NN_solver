#!/home/viera/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.misc import derivative
import os
import time

import matplotlib.pyplot as plt

from sympy import symbols, lambdify
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
import numpy as np

#torch.nn.functional.sigmoid

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear( 1, H)
		self.fc2 = nn.Linear(H, H)
		self.fc3 = nn.Linear(H, 1)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(self.fc2(x))
		#x = torch.sigmoid(self.fc2(x))
		x = self.fc3(x)
		return x

def Optimizer(x,y,x0,stop):
	loss_fn = torch.nn.MSELoss(reduction='sum')
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
	t=0
	while True:
		try:
			y_pred = net(x) + (x-x0)*derivative(net,x)
			loss = loss_fn(y_pred, y)

			if t % 100 == 99:
				print(t, loss.item())
			t += 1
			if t == stop:
				break
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		except KeyboardInterrupt:
			return loss.item()

def do_it(equation,initial_condition,evaluation,min,max,neuronas,iteraciones):
	global H, net, learning_rate
	H,learning_rate = neuronas,1e-4

	if torch.cuda.is_available():
		torch.cuda.device('cuda')

	net = Net()

	xs = symbols(equation[0][1])
	equation = equation[1]
	try:
		syms = parse_expr(equation)
	except:
		return 1

	func = lambdify(xs,syms)

	x0,y0 = initial_condition[0], initial_condition[1]
	datax = torch.linspace(min,max).view(-1,1)
	datay = func(datax)

	time_s = time.time()
	loss_ = Optimizer(datax,datay,x0,iteraciones)
	print('\n')
	print('T = '+str(time.time() - time_s)+' sec')

	func = lambda x: y0+(torch.Tensor([x])-x0)*net(torch.Tensor([x]))
	funcT = lambda x: y0+(x-x0)*net(x)

	ys = [func(x) for x in datax]

	plt.figure(len(os.listdir('img/')))
	plt.plot(datax.view(-1,1),ys)
	plt.savefig(f"img/fig{len(os.listdir('img/'))}.png")

	evaluation = round(float(func(evaluation)[0]),2)

	return evaluation
