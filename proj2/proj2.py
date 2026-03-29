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
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        for _ in range(self.n_iter):
            output = self.activation(self.net_input(X))
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict_proba(self, X):
        p1 = self.activation(self.net_input(X))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


def plot_decision_boundary(ax, clf, X, y, title):
    cmap = ListedColormap(['#FFAAAA', '#AAAAFF'])
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                            np.linspace(x2_min, x2_max, 200))
    Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', s=5, alpha=0.4)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', s=5, alpha=0.4)
    ax.set_title(title)


print("=== Custom Logistic Regression ===")
lr_custom = LogisticRegressionGD(eta=0.05, n_iter=200)
lr_custom.fit(X_train_std, y_train)
print(f"Train accuracy: {accuracy_score(y_train, lr_custom.predict(X_train_std)):.3f}")
print(f"Test accuracy:  {accuracy_score(y_test,  lr_custom.predict(X_test_std)):.3f}")

print("\n=== Decision Tree ===")
for criterion in ['entropy', 'gini']:
    for depth in [3, 5, 10, None]:
        dt = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        print(f"  {criterion}, depth={depth}: train={accuracy_score(y_train, dt.predict(X_train)):.3f}  test={accuracy_score(y_test, dt.predict(X_test)):.3f}")

print("\n=== Random Forest ===")
for n in [10, 50, 100, 200]:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    print(f"  n_estimators={n}: train={accuracy_score(y_train, rf.predict(X_train)):.3f}  test={accuracy_score(y_test, rf.predict(X_test)):.3f}")

print("\n=== SVM ===")
svm = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
svm.fit(X_train_std, y_train)
print(f"Train accuracy: {accuracy_score(y_train, svm.predict(X_train_std)):.3f}")
print(f"Test accuracy:  {accuracy_score(y_test,  svm.predict(X_test_std)):.3f}")

print("\n=== Voting Classifier ===")
lr_sk  = LogisticRegression(C=1.0, random_state=42)
rf_v   = RandomForestClassifier(n_estimators=100, random_state=42)
svm_v  = SVC(kernel='rbf', probability=True, random_state=42)

voting_hard = VotingClassifier(estimators=[('lr', lr_sk), ('rf', rf_v), ('svm', svm_v)], voting='hard')
voting_soft = VotingClassifier(estimators=[('lr', lr_sk), ('rf', rf_v), ('svm', svm_v)], voting='soft')

voting_hard.fit(X_train_std, y_train)
voting_soft.fit(X_train_std, y_train)

print(f"Hard voting - train={accuracy_score(y_train, voting_hard.predict(X_train_std)):.3f}  test={accuracy_score(y_test, voting_hard.predict(X_test_std)):.3f}")
print(f"Soft voting - train={accuracy_score(y_train, voting_soft.predict(X_train_std)):.3f}  test={accuracy_score(y_test, voting_soft.predict(X_test_std)):.3f}")


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

class Wrap:
    def __init__(self, fn): self.fn = fn
    def predict(self, X): return self.fn(X)

plot_decision_boundary(axes[0, 0], Wrap(lambda X: lr_custom.predict(scaler.transform(X))),   X_test, y_test, 'Custom Logistic Regression')
plot_decision_boundary(axes[0, 1], DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42).fit(X_train, y_train), X_test, y_test, 'Decision Tree (Gini, depth=5)')
plot_decision_boundary(axes[0, 2], RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train), X_test, y_test, 'Random Forest (100 trees)')
plot_decision_boundary(axes[1, 0], Wrap(lambda X: svm.predict(scaler.transform(X))),          X_test, y_test, 'SVM (RBF kernel)')
plot_decision_boundary(axes[1, 1], Wrap(lambda X: voting_hard.predict(scaler.transform(X))),  X_test, y_test, 'Voting Classifier (Hard)')
plot_decision_boundary(axes[1, 2], Wrap(lambda X: voting_soft.predict(scaler.transform(X))),  X_test, y_test, 'Voting Classifier (Soft)')

plt.tight_layout()
plt.savefig('decision_boundaries.png', dpi=120)
plt.show()
print("\nPlot saved.")