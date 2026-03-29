import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class LogisticRegressionGD:
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta          = eta
        self.n_iter       = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rng      = np.random.RandomState(self.random_state)
        self.w_  = rng.normal(0.0, 0.01, 1 + X.shape[1])
        for _ in range(self.n_iter):
            out       = self._sigmoid(self._net(X))
            err       = y - out
            self.w_[1:] += self.eta * X.T.dot(err)
            self.w_[0]  += self.eta * err.sum()
        return self

    def _net(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict_proba(self, X):
        p1 = self._sigmoid(self._net(X))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return np.where(self._net(X) >= 0.0, 1, 0)


def load_data():
    X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    return X_tr, X_te, y_tr, y_te, sc.fit_transform(X_tr), sc.transform(X_te), sc


def boundary_plot(ax, clf, X, y, title):
    cmap = ListedColormap(['#FFAAAA', '#AAAAFF'])
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                           np.linspace(x2_min, x2_max, 200))
    Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)
    ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red',  s=5, alpha=0.4)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', s=5, alpha=0.4)
    ax.set_title(title)


class ScaledWrap:
    def __init__(self, clf, sc):
        self.clf = clf
        self.sc  = sc
    def predict(self, X):
        return self.clf.predict(self.sc.transform(X))


def run():
    X_tr, X_te, y_tr, y_te, X_tr_s, X_te_s, sc = load_data()

    print("=== Custom Logistic Regression ===")
    lr_gd = LogisticRegressionGD(eta=0.05, n_iter=200)
    lr_gd.fit(X_tr_s, y_tr)
    print(f"  train={accuracy_score(y_tr, lr_gd.predict(X_tr_s)):.3f}  test={accuracy_score(y_te, lr_gd.predict(X_te_s)):.3f}")

    print("\n=== Decision Tree ===")
    for criterion in ['entropy', 'gini']:
        for depth in [3, 5, 10, None]:
            dt = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
            dt.fit(X_tr, y_tr)
            print(f"  {criterion}, depth={depth}: train={accuracy_score(y_tr, dt.predict(X_tr)):.3f}  test={accuracy_score(y_te, dt.predict(X_te)):.3f}")

    print("\n=== Random Forest ===")
    for n in [10, 50, 100, 200]:
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(X_tr, y_tr)
        print(f"  n={n}: train={accuracy_score(y_tr, rf.predict(X_tr)):.3f}  test={accuracy_score(y_te, rf.predict(X_te)):.3f}")

    print("\n=== SVM ===")
    svm = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
    svm.fit(X_tr_s, y_tr)
    print(f"  train={accuracy_score(y_tr, svm.predict(X_tr_s)):.3f}  test={accuracy_score(y_te, svm.predict(X_te_s)):.3f}")

    print("\n=== Voting Classifier ===")
    lr_sk = LogisticRegression(C=1.0, random_state=42)
    rf_v  = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_v = SVC(kernel='rbf', probability=True, random_state=42)

    v_hard = VotingClassifier([('lr', lr_sk), ('rf', rf_v), ('svm', svm_v)], voting='hard')
    v_soft = VotingClassifier([('lr', lr_sk), ('rf', rf_v), ('svm', svm_v)], voting='soft')
    v_hard.fit(X_tr_s, y_tr)
    v_soft.fit(X_tr_s, y_tr)
    print(f"  hard: train={accuracy_score(y_tr, v_hard.predict(X_tr_s)):.3f}  test={accuracy_score(y_te, v_hard.predict(X_te_s)):.3f}")
    print(f"  soft: train={accuracy_score(y_tr, v_soft.predict(X_tr_s)):.3f}  test={accuracy_score(y_te, v_soft.predict(X_te_s)):.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    dt_plot  = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42).fit(X_tr, y_tr)
    rf_plot  = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)

    boundary_plot(axes[0, 0], ScaledWrap(lr_gd,   sc), X_te, y_te, 'Custom Logistic Regression')
    boundary_plot(axes[0, 1], dt_plot,                  X_te, y_te, 'Decision Tree (Gini, depth=5)')
    boundary_plot(axes[0, 2], rf_plot,                  X_te, y_te, 'Random Forest (100 trees)')
    boundary_plot(axes[1, 0], ScaledWrap(svm,     sc),  X_te, y_te, 'SVM (RBF kernel)')
    boundary_plot(axes[1, 1], ScaledWrap(v_hard,  sc),  X_te, y_te, 'Voting Classifier (Hard)')
    boundary_plot(axes[1, 2], ScaledWrap(v_soft,  sc),  X_te, y_te, 'Voting Classifier (Soft)')

    plt.tight_layout()
    plt.savefig('decision_boundaries.png', dpi=120)
    plt.show()


if __name__ == "__main__":
    run()
