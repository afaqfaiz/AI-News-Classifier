
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # for numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


class LogisticRegressionScratch:
    def __init__(self, num_features, num_classes, learning_rate=0.1, epochs=1000):
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = np.zeros((num_features, num_classes))
        self.b = np.zeros((1, num_classes))

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def _cross_entropy(self, probs, Y):
        return -np.mean(np.sum(Y * np.log(probs + 1e-15), axis=1))

    def train(self, X_train, y_train, X_val, y_val):
        m = X_train.shape[0]
        Y_train = one_hot_encode(y_train, self.num_classes)
        Y_val = one_hot_encode(y_val, self.num_classes)

        for epoch in range(1, self.epochs + 1):
            # Forward pass (train)
            logits = np.dot(X_train, self.W) + self.b
            probs = softmax(logits)
            train_loss = self._cross_entropy(probs, Y_train)
            train_preds = np.argmax(probs, axis=1)
            train_acc = accuracy(y_train, train_preds)

            # Backward pass
            dW = (1 / m) * np.dot(X_train.T, (probs - Y_train))
            db = (1 / m) * np.sum(probs - Y_train, axis=0, keepdims=True)

            # Update parameters
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            # Validation
            val_logits = np.dot(X_val, self.W) + self.b
            val_probs = softmax(val_logits)
            val_loss = self._cross_entropy(val_probs, Y_val)
            val_preds = np.argmax(val_probs, axis=1)
            val_acc = accuracy(y_val, val_preds)

            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # Print every 50 epochs
            if epoch % 50 == 0 or epoch == 1:
                print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    def predict(self, X):
        logits = np.dot(X, self.W) + self.b
        probs = softmax(logits)
        return np.argmax(probs, axis=1)
