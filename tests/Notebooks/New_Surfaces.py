# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:28:50 2025

@author: scrogows
"""



import matplotlib.pyplot as plt
import numpy as np

from Modules.Utils.Imports import *
from Modules.Models.BuildBINNs import BINNCovasim
from Modules.Utils.ModelWrapper import ModelWrapper
import torch

import Modules.Utils.PDESolver as PDESolver
import Modules.Loaders.DataFormatter as DF
from utils import get_case_name, lasso_parameter_fitting
import seaborn as sns
# sns.set(font_scale=1.2, style='white')
from sklearn import linear_model
# helper functions
def to_torch(x):
    return torch.from_numpy(x).float().to(device)
def to_numpy(x):
    return x.detach().cpu().numpy()


device = torch.device(GetLowestGPU(pick_from=[0]))
# instantiate BINN
path = '../Data/covasim_data/'
population = 200000
test_prob = 0.1
trace_prob = 0.3
keep_d = True
retrain = False
dynamic = True
chi_type = 'sin'
case_name = get_case_name(population, test_prob, trace_prob, keep_d, dynamic=dynamic, chi_type=chi_type)

params = DF.load_covasim_data(path, population, test_prob, trace_prob, case_name, plot=False)
data = params['data']
data = (data / params['population']).to_numpy()
N = len(data)
t_max = N - 1
t = np.arange(N)
params.pop('data')
tracing_array = params['tracing_array']

#mydir = '../models/covasim/UPDATED/2024-11-20_16-58-20' #piecewise
#mydir = '../models/covasim/UPDATED/2024-11-20_20-00-41' #constant
mydir = '../models/covasim/UPDATED/2024-11-21_05-28-40' #sin

binn = BINNCovasim(params, t_max, tracing_array, keep_d=keep_d).to(device)
parameters = binn.parameters()
model = ModelWrapper(binn, None, None, save_name=os.path.join(mydir, case_name))

# load model weights
# model.save_name = '../Weights/'
# model.save_name += case_name
if retrain:
    model.save_name += '_retrain'
model.save_name += '_best_val'
model.load(model.save_name + '_model', device=device)
save_path = model.save_folder
# grab initial condition
u0 = data[0, :].copy()

# grab value ranges
yita_lb, yita_ub = model.model.yita_lb, model.model.yita_ub
beta_lb, beta_ub = model.model.beta_lb, model.model.beta_ub
tau_lb, tau_ub = model.model.tau_lb, model.model.tau_ub

# learned contact_rate function
def contact_rate(u):
    res = binn.eta_func(to_torch(u)) # [:,[0,3,4]]
    return to_numpy(res)

def beta(u):
    res = binn.beta_func(to_torch(u))
    return to_numpy(res)

def tau(u):
    res = binn.tau_func(to_torch(u))
    return to_numpy(res)

x = data[:,0]
y = data[:,3]
z = data[:,4]
u_grid = np.stack([np.ravel(x), np.ravel(y), np.ravel(z)], axis=1)
res1 = contact_rate(u_grid)
res1 = yita_lb + (yita_ub - yita_lb) * res1  # scaling
res1 = res1[:,0].reshape(x.shape)
res1 = np.round(res1, decimals=6)
res1 = res1.astype(float)
#labels = ['S', 'A', 'Y', 'Eta']


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the scatter points
# #ax.plot_trisurf(x, y, res, cmap=cm.coolwarm, alpha=1)
# ax.scatter(x, y, res1, c=z, s=10, alpha=0.8)
# ax.set_xlabel(labels[0])
# ax.set_ylabel(labels[1])
# ax.set_zlabel(labels[3])
# plt.show()
# plt.scatter(z, res, label='eta', color='blue', marker='o', s=20)
# plt.xlabel('Y')
# plt.ylabel('Eta')
# plt.show()

###Original Surfaces###
s_min, s_max = data[:,0].min(), data[:,0].max()
# a_min, a_max = 0.0, 0.015 # data[:,3].min(), data[:,3].max()
# y_min, y_max = 0.0, 0.015 # data[:,4].min(), data[:,4].max()
a_min, a_max = data[:,3].min(), data[:,3].max()
y_min, y_max = data[:,4].min(), data[:,4].max()
# chi_min, chi_max = 0.0, params['eff_ub']

a_grid = np.linspace(s_min, s_max, 10)
b_grid = np.linspace(a_min, a_max, 10)
c_grid = np.linspace(y_min, y_max, 10)
labels = ['S', 'A', 'Y']
# for 3 inputs
fig = plt.figure(figsize=(10,7))
for i in range(3):
    if i == 0:
        X, Y = np.meshgrid(a_grid, b_grid)
        Z = np.ones_like(X) * c_grid.mean()
        x_label, y_label = labels[1], labels[0]
    elif i == 1:
        X, Z = np.meshgrid(a_grid, c_grid)
        Y = np.ones_like(X) * b_grid.mean()
        x_label, y_label = labels[2], labels[0]
    else:
        Y, Z = np.meshgrid(b_grid, c_grid)
        X = np.ones_like(Y) * a_grid.mean()
        #x_label, y_label = labels[1], labels[2]
        x_label, y_label = labels[2], labels[1] #switching axis
    u_grid = np.stack([np.ravel(X), np.ravel(Y), np.ravel(Z)], axis=1)
    res = contact_rate(u_grid)
    res = yita_lb + (yita_ub - yita_lb) * res  # scaling
    res = res[:,0].reshape(X.shape)
    res = np.round(res, decimals=6)
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    if i == 0:
        ax.scatter(y, x, res1, c=z, s=15, alpha=1)
        ax.plot_surface(Y, X, res, cmap=cm.coolwarm, alpha=0.5)
        ax.scatter(Y.reshape(-1), X.reshape(-1), res.reshape(-1), s=5, c='k')
        
    elif i == 1:
        
        ax.plot_surface(Z, X, res, cmap=cm.coolwarm, alpha=0.5)
        ax.scatter(Z.reshape(-1), X.reshape(-1), res.reshape(-1), s=5, c='k')
        ax.scatter(z, x, res1, c=z, s=15, alpha=1)
    else:
        #ax.plot_surface(Y, Z, res, cmap=cm.coolwarm, alpha=1)
        ax.scatter(z, y, res1, c=z, s=15, alpha=1)
        ax.plot_surface(Z, Y, res, cmap=cm.coolwarm, alpha=0.5) #switching axis
        #ax.scatter(Y.reshape(-1), Z.reshape(-1), res.reshape(-1), s=5, c='k')
        ax.scatter(Z.reshape(-1), Y.reshape(-1), res.reshape(-1), s=2, c='k') #switching axis
        
        plt.setp(ax.get_xticklabels(), rotation=15) # , ha="right", rotation_mode="anchor"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

plt.tight_layout(pad=2)
plt.show()
pass
#%% visualization for beta
chi_min, chi_max = params['tracing_array'].min(), params['tracing_array'].max()

SAY = data[:,0] + data[:,3] + data[:,4]
SAY_min, SAY_max = SAY.min(), SAY.max()
chi = params['tracing_array']

a_grid = np.linspace(SAY_min, SAY_max, 10) # D + R + F
b_grid = np.linspace(chi_min, chi_max, 10)
labels = ['S + A + Y', r'$h(t)$']

###Beta from Data##
u_grid1 = np.stack([np.ravel(SAY), np.ravel(chi)], axis=1)
res1 = beta(u_grid) * params['n_contacts'] # * u_grid[:, [1]] *
res1 = res1[:,0].reshape(SAY.shape)
res1 = np.round(res1, decimals=6)

##Beta interpolation from Grid###
X, Y = np.meshgrid(a_grid, b_grid)
x_label, y_label = labels[0], labels[1]
u_grid = np.stack([np.ravel(X), np.ravel(Y)], axis=1)
res = beta(u_grid) * params['n_contacts'] # * u_grid[:, [1]] *
res = res[:,0].reshape(X.shape)
res = np.round(res, decimals=6)
# res = beta_lb + (beta_ub - beta_lb) * res

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(SAY, chi, res1, c=z, s=15, alpha=1)
ax.plot_surface(X, Y, res, cmap=cm.coolwarm, alpha=0.5)
ax.scatter(X.reshape(-1), Y.reshape(-1), res.reshape(-1), s=5, c='k')
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
# ax.set_zticks(np.arange(0,10.1, 2), np.arange(0,10.1, 2))
# ax.set_title(r'$\beta(t)$')
plt.show()
plt.tight_layout(pad=2)
plt.savefig(os.path.join(save_path, case_name + '_parameter_NN_beta' + '.png'), dpi=300, bbox_inches='tight' )
plt.close()

# # plot again for real scale of compartment
# fig = plt.figure(figsize=(10,7))
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.plot_surface(X, Y, res, cmap=cm.coolwarm, alpha=1)
# ax.scatter(X.reshape(-1), Y.reshape(-1), res.reshape(-1), s=5, c='k')
# ax.set_xticks(a_grid, np.int_(a_grid * population))
# ax.set_xlabel(x_label)
# ax.set_ylabel(y_label)
# plt.tight_layout(pad=2)
# plt.savefig(os.path.join(save_path, case_name + '_parameter_NN_beta_scale' + '.png') )
# plt.close()
pass
#%% visualization for tau

a_grid = np.linspace(a_min, a_max, 10)
b_grid = np.linspace(y_min, y_max, 10)
labels = ['A', 'Y']

X, Y = np.meshgrid(a_grid, b_grid)
x_label, y_label = labels[0], labels[1]
u_grid = np.stack([np.ravel(X), np.ravel(Y)], axis=1)
res = tau(u_grid)
res = res[:,0].reshape(X.shape)
res = tau_lb + (tau_ub - tau_lb) * res # scaling
res = np.round(res, decimals=4)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(X, Y, res, cmap=cm.coolwarm, alpha=1)
ax.scatter(X.reshape(-1), Y.reshape(-1), res.reshape(-1), s=5, c='k')
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
plt.tight_layout(pad=2)
plt.savefig(os.path.join(save_path, case_name + '_parameter_NN_tau' + '.png'), dpi=300, bbox_inches='tight') #
plt.close()

# for 3 inputs
# fig = plt.figure(figsize=(10,7))
# for i in range(3):
#     if i == 0:
#         X, Y = np.meshgrid(a_grid, b_grid)
#         Z = np.ones_like(X) * chi_max
#         x_label, y_label = labels[0], labels[1]
#     elif i == 1:
#         X, Z = np.meshgrid(a_grid, c_grid)
#         Y = np.ones_like(X) * b_grid.mean()
#         x_label, y_label = labels[0], labels[2]
#     else:
#         Y, Z = np.meshgrid(b_grid, c_grid)
#         X = np.ones_like(Y) * a_grid.mean()
#         x_label, y_label = labels[1], labels[2]
#     u_grid = np.stack([np.ravel(X), np.ravel(Y), np.ravel(Z)], axis=1)
#     res = beta(u_grid)
#     res = beta_lb + (beta_ub - beta_lb) * res
#     res = res[:,0].reshape(X.shape)
#     ax = fig.add_subplot(1, 3, i + 1, projection='3d')
#     if i == 0:
#         ax.plot_surface(X, Y, res, cmap=cm.coolwarm, alpha=1)
#         ax.scatter(X.reshape(-1), Y.reshape(-1), res.reshape(-1), s=5, c='k')
#     elif i == 1:
#         ax.plot_surface(X, Z, res, cmap=cm.coolwarm, alpha=1)
#         ax.scatter(X.reshape(-1), Z.reshape(-1), res.reshape(-1), s=5, c='k')
#     else:
#         ax.plot_surface(Y, Z, res, cmap=cm.coolwarm, alpha=1)
#         ax.scatter(Y.reshape(-1), Z.reshape(-1), res.reshape(-1), s=5, c='k')
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
# plt.tight_layout(pad=2)
# plt.savefig(os.path.join(save_path, case_name + '_parameter_NN_beta' + '.png') )
# plt.close()
# lasso regression to learn symbolic terms from parameter NN
def get_samples_ct(u):
    s, a, y =  u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None]
    candidates = [s, s**2, a, y] # s related terms
    # candidates += [a]
    # candidates += [y]
    # candidates += [chi]
    candidates = np.concatenate(candidates, axis=1)
    return candidates

def get_samples_beta(u):
    drf, chi = u[:, 0][:, None], u[:, 1][:, None]
    candidates = [drf, chi] # , chi**2
    candidates = np.concatenate(candidates, axis=1)
    return candidates

def get_samples_tau(u):
    a, y = u[:, 0][:, None], u[:, 1][:, None]
    candidates = [a, y]
    candidates = np.concatenate(candidates, axis=1)
    return candidates

s_grid = np.linspace(s_min, s_max, 10)
a_grid = np.linspace(a_min, a_max, 10)
y_grid = np.linspace(y_min, y_max, 10)
train_x = np.array(np.meshgrid(s_grid, a_grid, y_grid)).T.reshape(-1,3)
data_x = get_samples_ct(train_x)
data_y = contact_rate(train_x)
data_y = data_y[:,0][:, None]
# data_y = yita_lb + (yita_ub - yita_lb) * data_y

term_names = ['S', 'S^2', 'A', 'Y'] #
lasso_parameter_fitting(data_x, data_y, 'eta', save_path, case_name, True, term_names)


a_grid = np.linspace(0.6, 1.0, 10) # D + R + F
b_grid = np.linspace(chi_min, chi_max, 10)
term_names = ['S + A + Y', r'$\chi$']
# c_grid = np.ones_like(a_grid) * chi_max
train_x = np.array(np.meshgrid(a_grid, b_grid)).T.reshape(-1,2)
data_x = get_samples_beta(train_x)
data_y = beta(train_x)
data_y = data_y[:,0][:, None]
lasso_parameter_fitting(data_x[:, :], data_y, 'beta', save_path, case_name, True, term_names)

a_grid = np.linspace(a_min, a_max, 10)
b_grid = np.linspace(y_min, y_max, 10)
term_names = ['A', 'Y']
# c_grid = np.ones_like(a_grid) * chi_max
train_x = np.array(np.meshgrid(a_grid, b_grid)).T.reshape(-1,2)
data_x = get_samples_tau(train_x)
data_y = tau(train_x)
data_y = data_y[:,0][:, None]
# data_y = tau_lb + (tau_ub - tau_lb) * data_y
lasso_parameter_fitting(data_x[:, :], data_y, 'tau', save_path, case_name, True, term_names)