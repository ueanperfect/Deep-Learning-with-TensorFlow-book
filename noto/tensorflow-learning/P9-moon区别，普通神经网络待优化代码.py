from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


N_SAMPLES = 2000
X,y=make_moons(n_samples=N_SAMPLES,noise=0.25,random_state=100)
X_train ,X_test,y_train,y_test = train_test_split(X,y,test_size = 400,random_state=42)

print(X_train)
print(X_test)
print(y_test)

def make_plot(X, y, plot_name, file_name, XX=None, YY=None, preds=None):
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([x_min,x_max])
    axes.set_ylim([y_min, y_max])
    axes.set(xlabel="$x_1$", ylabel="$x_2$")
    if (XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25,alpha=0.08,cmap=cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5],cmap="Greys",vmin=0, vmax=.6)
        markers = ['o' if i == 1 else 's' for i in y.ravel()]
        # 绘制正负样本
        mscatter(X[:, 0], X[:, 1], c=y.ravel(), s=20, cmap=plt.cm.Spectral, edgecolors='none', m=markers)
        plt.savefig(OUTPUT_DIR + '/' + file_name)
        make_plot(X, y, None, "dataset.svg")