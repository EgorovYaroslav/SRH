import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


class EnsembleClassifier(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=2):  # Now input_dim=6
        super(EnsembleClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.classifier(x)

    def prepare_input(self, effnet_probs, catboost_probs, clip_probs):
        """
        Combine probabilities from all three models.
        Args:
            effnet_probs (np.ndarray): Shape (N, 2)
            catboost_probs (np.ndarray): Shape (N, 2)
            clip_probs (np.ndarray): Shape (N, 2)
        Returns:
            inputs (np.ndarray): Combined shape (N, 6)
        """
        return np.hstack([effnet_probs, catboost_probs, clip_probs])

    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, lr=1e-3):
        """
        Train the ensemble model with a progress bar using tqdm.
        Args:
            X_train (np.ndarray): Shape (N, 6)
            y_train (np.ndarray): Shape (N,)
        """
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        device = next(self.parameters()).device
        self.to(device)
        self.train()

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                acc = correct / total
                progress_bar.set_postfix(loss=running_loss/total, accuracy=acc)

            if X_val is not None and y_val is not None:
                val_acc = self.evaluate(X_val, y_val)
                print(f"Validation Accuracy: {val_acc:.4f}")

    def predict(self, X):
        """
        Predict class labels.
        Args:
            X (np.ndarray): Shape (N, 6)
        Returns:
            preds (np.ndarray): Shape (N,)
        """
        self.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(next(self.parameters()).device)
            outputs = self(inputs)
            _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities.
        Args:
            X (np.ndarray): Shape (N, 6)
        Returns:
            probs (np.ndarray): Shape (N, 2)
        """
        self.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(next(self.parameters()).device)
            probs = self(inputs)
        probs = probs.cpu().numpy()
        return probs  # Return full probability distribution per sample

    def evaluate(self, X_test, y_test):
        """
        Evaluate accuracy on test data.
        """
        preds = self.predict(X_test)
        acc = np.mean(preds == y_test)
        return acc

    def get_confusion_matrix(self, X_test, y_test, normalize=True):
        """
        Generate a confusion matrix based on test data.
        Args:
            X_test (np.ndarray): Input features, shape (N, 6)
            y_test (np.ndarray): True labels, shape (N,)
            normalize (bool): Whether to normalize values to percentages
        Returns:
            cm (np.ndarray): Confusion matrix
        """
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        return cm

    def confusion_matrix(self, X_test, y_test, filename='ensemble', save_pdf=True, show_plot=True):
        """
        Plot and optionally save the confusion matrix.
        Args:
            X_test (np.ndarray): Input features, shape (N, 6)
            y_test (np.ndarray): True labels, shape (N,)
            filename (str): Output filename for PDF
            save_pdf (bool): Whether to save as PDF
            show_plot (bool): Whether to display plot
        """
        cm = self.get_confusion_matrix(X_test, y_test, normalize=True)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=['Bad', 'Ok'],
                    yticklabels=['Bad', 'Ok'])

        plt.title('Confusion Matrix - Ensemble Classifier')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        if save_pdf:
            pdf_path = f"fig/{filename}.pdf"
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(plt.gcf(), bbox_inches='tight')

        if show_plot:
            plt.show()
        else:
            plt.close()

    def save_model(self, path='ensemble_model.pth'):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path='ensemble_model.pth', input_dim=6, hidden_dim=64, output_dim=2):
        model = cls(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"Model loaded from {path}")
        return model

    def get_metrics(self, X_test, y_test,):
        from sklearn.metrics import classification_report

        y_pred = self.predict(X_test)

        report = classification_report(
            y_test,
            y_pred,
            target_names=['Bad', 'Ok'],
            output_dict=True
        )

        accuracy = report['accuracy']
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score
        }