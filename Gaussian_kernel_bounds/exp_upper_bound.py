#!/usr/bin/env python

import numpy as np
import sys
import sklearn.metrics
import matplotlib
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

σ = 0.5
c = (1/(σ*σ*(1-np.exp(-1/(σ*σ)))))
xₒ = np.sqrt(2)

x = np.arange(-2,2,0.1)
y1 = np.exp(-(x*x)/(2*σ*σ))
y2 = 1 - (x*x)/(2*σ*σ*c)

fₐ_xₒ = np.exp(-(xₒ*xₒ)/(2*σ*σ))
fᵦ_xₒ = 1 - (xₒ*xₒ)/(2*σ*σ*c)

print(fₐ_xₒ)
print(fᵦ_xₒ)

plt.plot(x, y1, 'r-');
plt.plot(x, y2, 'b-');
plt.ylim(0, 1.5)
plt.axvline(np.sqrt(2), linestyle='--')
plt.axvline(-np.sqrt(2), linestyle='--')
plt.title('Gaussian Upper Bound within $\sqrt{2}$')
plt.rcParams.update({'font.size': 18})
plt.text(-1.0,1.3,r'Blue : $1 - \frac{x^2}{2c\sigma^2}, c = \frac{1}{\sigma^2 (1 - e^{-\frac{1}{\sigma^2}})}$')
plt.text(-1.0,1.1,r'Red : $e^{-\frac{x^2}{2\sigma^2}}$')
#plt.text(-1.0,0.9,r'$c = \frac{1}{\sigma^2 (1 - e^{-\frac{1}{\sigma^2}})}$')
#plt.arrow(np.sqrt(2) - 1, 0.0183 + 0.2, 1, -0.2, length_includes_head=True, label='hello')
arrow_properties = dict( facecolor="black", width=0.8, headwidth=4, shrink=0.1)
msg = r'$(\sqrt{2}, 0.0183) , 1 - \frac{x^2}{2c\sigma^2} = e^{-\frac{x^2}{2\sigma^2}}$'
plt.annotate(msg, xy=(np.sqrt(2), 0.0183), xytext=(np.sqrt(2)-3.5, 0.0183 + 0.2), arrowprops=arrow_properties)

plt.show()

