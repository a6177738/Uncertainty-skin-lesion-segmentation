import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
fig= plt.figure()
ax = fig.add_subplot(111)
ax.set(xlim= [0,0.5],ylim=[0,1],title='The relationship between thresholds and initial pseudo-Label accuracy.',ylabel='ACC',xlabel='threshold')
x= [0,0.1,0.2,0.3,0.4,0.5]


#y_labeling = np.array([0, 0.0534,0.1903 , 0.4248, 0.7291, 1])
y_labelacc = np.array([ 83.87/100, 77.37/100, 86.82/100, 90.62/100, 90.12/100, 76.63/100])
y_label0_5acc = np.array([0.1939, 24.28/100, 33.42/100, 50.93/100, 74.08/100, 76.63/100])
y_label5_10acc = np.array([0.7203,72.79/100, 75.56/100, 76.32/100, 79.84/100, 76.63/100])
# y_labeldic = np.array([84.91/100,66.79/100, 75.41/100, 77.51/100, 74.59/100, 52.13/100])
# y_label0_5dic = np.array([29.26/100,28.20/100, 29.83/100, 35.25/100, 46.77/100, 52.13/100])
# y_label5_10dic = np.array([0.0819/100, 50.86/100, 62.95/100, 67.78/100, 52.13/100])



ax.plot(x,y_label0_5acc,'b-o',label='threshold=x ',alpha=0.5)
ax.plot(x,y_label5_10acc,'g-*',label='threshold=1-x ',alpha=0.5)
ax.plot(x,y_labelacc,'y-P',label='our method',alpha=0.5)
#ax.plot(x,y_labeldic,'b-o',label='pseudo-label DIC',alpha=0.5)


plt.legend(loc='lower right')
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.show()
#fig.savefig("SISD_well.png")
fig.savefig("retios.png")