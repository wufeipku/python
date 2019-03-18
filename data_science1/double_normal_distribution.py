# imports
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import numpy as np
# setup plotly credentials
plotly.tools.set_credentials_file(username='csaurav', api_key='')
value_range = 2
xx = np.sort(np.random.normal(0, 1, 300)) # draw xx from a normal distribution
yy = np.sort(np.random.normal(0, 1, 300)) # drwa yy from a normal distribution
x,y=np.meshgrid(xx, yy)
mu_x = np.mean(xx)
sd_x = np.std(xx)
mu_y = np.mean(yy)
sd_y = np.std(yy)
rho = 0.5 # the expected correlation
# the formula to generate the pdf given X, Y
z = ((np.exp(-(1 / (2 * (1 - rho ** 2)))*(((x - mu_x) / sd_x) ** 2 + ((y - mu_y) / sd_y) ** 2
 - 2 * rho * (x - mu_x) * (y - mu_y)/(sd_x * sd_y))))
 / (2 * np.pi * sd_x * sd_y * np.sqrt(1 - rho ** 2)))
init_notebook_mode(connected=True)
colorscale = [[0.0, 'rgb(20,29,67)'],
 [0.1, 'rgb(28,76,96)'],
 [0.2, 'rgb(16,125,121)'],
 [0.3, 'rgb(92,166,133)'],
 [0.4, 'rgb(182,202,175)'],
 [0.5, 'rgb(253,245,243)'],
 [0.6, 'rgb(230,183,162)'],
 [0.7, 'rgb(211,118,105)'],
 [0.8, 'rgb(174,63,95)'],
 [0.9, 'rgb(116,25,93)'],
 [1.0, 'rgb(51,13,53)']]
textz = [['IQ: '+'{:0.5f}'.format(x[i][j])+'<br>Success: '+'{:0.5f}'.format(y[i][j]) +
 '<br>z: '+'{:0.5f}'.format(z[i][j]) for j in range(z.shape[1])] for i in range(z.shape[0])]
trace1 = go.Surface(
 x=tuple(x),
 y=tuple(y),
 z=tuple(z),
 colorscale=colorscale,
 text=textz,
 hoverinfo='text',
)
axis = dict(
 showbackground=True,
 backgroundcolor="rgb(230, 230,230)",
 showgrid=False,
 zeroline=False,
 showline=False)
layout = go.Layout(title="Bi-variate Probabilty Distribution Function",
 autosize=False,
 width=700,
 height=600,
 scene=dict(xaxis=dict(axis, range=[-1 * value_range, value_range],
 title='IQ'),
 yaxis=dict(
 axis, range=[-1 * value_range, value_range],
 title='Success'),
 zaxis=dict(
 axis, range=[np.min(z) - (np.max(z) - np.min(z)), np.max(z)],
 title='Probability P("Success", "IQ")'),
 aspectratio=dict(x=1,
 y=1,
 z=0.95)
 )
 )
z_offset = (np.min(z) - (np.max(z) - np.min(z)))*np.ones(z.shape)
x_offset = -2*np.ones(z.shape)
y_offset = -2*np.ones(z.shape)
def proj_z(x, y, z): return z # projection in the z-direction
colorsurfz = proj_z(x, y, z)
def proj_x(x, y, z): return x
colorsurfx = proj_z(x, y, z)
def proj_y(x, y, z): return y
colorsurfy = proj_z(x, y, z)
textx = [['Success: '+'{:0.5f}'.format(y[i][j])+'<br>z: '+'{:0.5f}'.format(z[i][j]) +
 '<br>IQ: '+'{:0.5f}'.format(x[i][j]) for j in range(z.shape[1])] for i in range(z.shape[0])]
texty = [['IQ: '+'{:0.5f}'.format(x[i][j])+'<br>z: '+'{:0.5f}'.format(z[i][j]) +
 '<br>Success: '+'{:0.5f}'.format(y[i][j]) for j in range(z.shape[1])] for i in range(z.shape[0])]
tracex = go.Surface(z=list(z),
 x=list(x_offset),
 y=list(y),
 colorscale=colorscale,
 showlegend=False,
 showscale=False,
 surfacecolor=colorsurfx,
 text=textx,
 hoverinfo='text'
 )
tracey = go.Surface(z=list(z),
 x=list(x),
 y=list(y_offset),
 colorscale=colorscale,
 showlegend=False,
 showscale=False,
 surfacecolor=colorsurfy,
 text=texty,
 hoverinfo='text'
 )
tracez = go.Surface(z=list(z_offset),
 x=list(x),
 y=list(y),
 colorscale=colorscale,
 showlegend=False,
 showscale=False,
 surfacecolor=colorsurfx,
 text=textz,
 hoverinfo='text'
 )
data = [trace1, tracex, tracey, tracez]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
plotly.tools.set_credentials_file(username='csaurav', api_key='')
py.plot(fig)
mean = (0, 0) # Mean for both success and IQ is zero
cov = [[1, .5], [.5, 1]] # covarnance matrix with assumption of sd =1 (any other sd gives the same result)
x = np.random.multivariate_normal(mean, cov, 10000) # we draw from both IQ and Success from a Multivariate
count_both = 0
count_pos_iq = 0
for i in range(len(x)):
 if (x[i, 0] > 0):
     count_pos_iq += 1
     if (x[i, 1] > 0):
        count_both += 1
     p = p + 1
count_both/count_pos_iq # ration of values where Succuess > 0, IQ >0 to those where IQ > 0