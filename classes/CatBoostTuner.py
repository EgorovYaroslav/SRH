class CatBoostTuner:
    def __init__(self, embeddings, classes=[], name='default', labels=['Bad', 'Ok']):
        self.name = name
        self.labels = labels
        self.embeddings = embeddings
        self.classes = classes

    def load_model(self):
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        return model.load_model(f'models/catboost_model_{self.name}')

    def train(self):
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(iterations=2000,
                                   task_type="GPU",
                                   devices='0')
        model.fit(self.embeddings,
                  self.classes,
                  verbose=False)
        model.save_model(f'models/catboost_model_{self.name}')

    def predict(self, prediction_type='Class'):
        import numpy as np
        model = self.load_model()
        y_pred = model.predict(self.embeddings, prediction_type)
        y_pred = np.atleast_2d(y_pred)
        if prediction_type == 'Class':
            return y_pred
        return np.argmax(y_pred, axis=1), np.max(y_pred, axis=1)

    def predict_probs(self, prediction_type='Probability'):
        model = self.load_model()
        y_pred = model.predict(self.embeddings, prediction_type)
        return y_pred

    def make_prediction(self):

        model = self.load_model()
        y_pred = model.predict(self.embeddings)
        return self.classes, y_pred

    def evaluate(self):
        from sklearn.metrics import f1_score

        y_test, y_pred = self.make_prediction()
        print(f"F1 = {f1_score(y_test, y_pred, average='micro'):.3f}")

    def get_confusion_matrix(self):
        import numpy as np
        from sklearn.metrics import confusion_matrix

        y_test, y_pred = self.make_prediction()
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        return cm

    def confusion_matrix(self, filename='catboost'):
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        import seaborn as sns

        pdf = PdfPages(f"fig/{filename}.pdf")
        figure = plt.figure(figsize=(10, 10))
        cm = self.get_confusion_matrix()
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.labels, yticklabels=self.labels,
                    annot_kws={'size': 16})
        plt.title('Confusion Matrix', fontsize=18)
        plt.xlabel('Predicted', fontsize=16)
        plt.ylabel('True', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        pdf.savefig(figure)
        pdf.close()

    def get_metrics(self):
        from sklearn.metrics import classification_report

        y_test, y_pred = self.make_prediction()

        report = classification_report(
            y_test,
            y_pred,
            target_names=self.labels,
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