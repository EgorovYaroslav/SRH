import os
import torch
import clip
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from tqdm import tqdm
import shutil


class ClipClassifier:
    def __init__(self, ds,dir_path='data/3Ghz', labels = {"Bad": "noise", "Ok": "circle"}, model_name="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the ClipClassifier.
        Args:
            dir_path (str): Path to the directory containing images.
            model_name (str): Name of the CLIP model to use.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.dir_path = f'{dir_path}/{ds}'
        self.ds = ds
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)

        # Folder names in the directory
        self.folder_names = sorted(os.listdir(self.dir_path))  # Assuming subfolders are 'Bad' and 'Ok'

        # Map folder names to CLIP-friendly class names
        self.class_mapping = labels
        self.class_names = [self.class_mapping[folder] for folder in self.folder_names]

        # Create text prompts for CLIP
        self.class_prompts = [f"a photo of {cls}" for cls in self.class_names]
        self.text_features = self._encode_text(self.class_prompts)

    def _encode_text(self, prompts):
        """
        Encode text prompts into features using CLIP.
        Args:
            prompts (list of str): List of text prompts.
        Returns:
            torch.Tensor: Text features.
        """
        with torch.no_grad():
            text_tokens = clip.tokenize(prompts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_embeddings(self, image_paths):
        """
        Get image embeddings for a list of image paths.
        Args:
            image_paths (list of str): List of paths to images.
        Returns:
            torch.Tensor: Image embeddings.
        """
        # image_paths = os.listdir(self.dir_path + "/" + folder)
        # image_dir = self.dir_path + "/" + folder
        image_embeddings = []
        for img_path in tqdm(image_paths):
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_features = self.model.encode_image(img_tensor)
                img_features /= img_features.norm(dim=-1, keepdim=True)
            image_embeddings.append(img_features.cpu())
        return torch.cat(image_embeddings, dim=0)

    def list_image_files(data_dir):
        import os

        # Supported image extensions
        image_extensions = ('.png', '.jpg', '.jpeg')

        # Get all files in the folder and filter by extension
        files = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.lower().endswith(image_extensions)
        ]

        return files
    def save_embeddings(self):

        image_paths = self.list_image_files(self.dir_path+"/Bad")
        embeddings0 = self.get_embeddings(image_paths)
        labels0 = [0] * len(image_paths)

        image_paths = self.list_image_files(self.dir_path+"/Ok")
        embeddings1 = self.get_embeddings(image_paths)
        labels1 = [1] * len(image_paths)

        embeddings = torch.cat((embeddings0, embeddings1), dim=0)
        labels = labels0 + labels1
        np.savez(f'embeddings/clip_embeddings_{self.ds}.npz', embeddings=embeddings, labels=labels)


    def classify(self, image_paths):
        """
        Classify a list of images.
        Args:
            image_paths (list of str): List of paths to images.
        Returns:
            list of str: Predicted class labels.
        """

        image_embeddings = self.get_embeddings(image_paths).to(self.device)
        with torch.no_grad():
            similarity = (image_embeddings @ self.text_features.T).cpu().numpy()

        # Get predicted class indices and probabilities using softmax
        logits = similarity  # Similarity scores can be treated as logits
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # For numerical stability
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # Softmax probabilities

        predicted_indices = np.argmax(similarity, axis=1)
        preds = [self.class_names[idx] for idx in predicted_indices]
        confidences = [probs[i, predicted_indices[i]] for i in range(len(predicted_indices))]

        return preds, confidences

    def evaluate(self):
        """
        Evaluate the classifier on all images in the directory.
        Returns:
            true_labels (list of str): Ground truth labels.
            predicted_labels (list of str): Predicted labels.
        """
        image_paths, true_labels = [], []
        for folder in self.folder_names:
            folder_dir = os.path.join(self.dir_path, folder)
            mapped_label = self.class_mapping[folder]  # Map folder name to CLIP-friendly class name
            for img_name in os.listdir(folder_dir):
                image_paths.append(os.path.join(folder_dir, img_name))
                true_labels.append(mapped_label)  # Use mapped label
        # Classify images
        predicted_labels, probabilities = self.classify(image_paths)
        return true_labels, predicted_labels, probabilities

    def save_evaluation(self):

        true_labels, predicted_labels, probabilities = self.evaluate()
        evaluation_data = {
            "true_labels": true_labels,
            "predicted_labels": predicted_labels,
            "probabilities": probabilities
        }

        with open(f"save/clip_classifier_{self.ds}.pkl", "wb") as f:
            pickle.dump(evaluation_data, f)
        print("Evaluation results saved")



    def load_evaluation(self):
        with open(f"save/clip_classifier_{self.ds}.pkl", "rb") as f:
            evaluation_data = pickle.load(f)
        true_labels = evaluation_data["true_labels"]
        predicted_labels = evaluation_data["predicted_labels"]
        probabilities = evaluation_data["probabilities"]
        return true_labels, predicted_labels, probabilities

    def get_confusion_matrix(self):
        if os.path.exists(f"save/clip_classifier_{self.ds}.pkl"):
            true_labels, predicted_labels, _ = self.load_evaluation()
        else:
            true_labels, predicted_labels, _ = self.evaluate()

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=self.class_names)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Normalize to percentages
        return cm

    def confusion_matrix(self, filename='clip'):
        from matplotlib.backends.backend_pdf import PdfPages

        cm = self.get_confusion_matrix()

        pdf = PdfPages(f"fig/{filename}.pdf")
        # Plot confusion matrix
        figure = plt.figure(figsize=(10, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            annot_kws={"size": 16},
        )
        plt.title("Confusion Matrix", fontsize=18)
        plt.xlabel("Predicted", fontsize=16)
        plt.ylabel("True", fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

        pdf.savefig(figure)
        pdf.close()

    def get_metrics(self):
        """
        Calculate Accuracy, Precision, Recall, and F1 Score.
        Returns:
            dict: A dictionary containing Accuracy, Precision, Recall, and F1 Score.
        """
        if os.path.exists(f"save/clip_classifier_{self.ds}.pkl"):
            true_labels, predicted_labels, _ = self.load_evaluation()
        else:
            true_labels, predicted_labels, _ = self.evaluate()

        # Generate classification report
        report = classification_report(
            true_labels,
            predicted_labels,
            target_names=self.class_names,
            output_dict=True
        )

        # Extract weighted averages for overall metrics
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

    def classify_and_organize(self, input_folder=None, output_folder="classified"):
        """
        Classify all images in a folder and organize them into subfolders:
            - 'Ok' for predictions matching 'circle'
            - 'Bad' for predictions matching 'noise'

        Args:
            input_folder (str): Path to folder containing images (if None, uses self.dir_path)
            output_folder (str): Path to save organized images
        """

        # Set input folder
        if input_folder is None:
            input_folder = self.dir_path  # Use dataset path if none provided

        # Create output directories
        os.makedirs(os.path.join(output_folder, "Ok"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "Bad"), os.makedirs(os.path.join(output_folder, "Unknown"), exist_ok=True))

        # Get image paths
        image_paths = []
        for root, _, files in os.walk(input_folder):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.fit', '.fits')):
                    image_paths.append(os.path.join(root, fname))

        print(f"Found {len(image_paths)} files to classify")

        # Process in batches
        batch_size = 32
        total = len(image_paths)

        for i in tqdm(range(0, total, batch_size), desc="Classifying and Organizing"):
            batch_paths = image_paths[i:i+batch_size]
            embeddings_batch = []

            # Load and preprocess images
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert("RGB")
                    image = self.preprocess(image).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        features = self.model.encode_image(image)
                        features /= features.norm(dim=-1, keepdim=True)
                        embeddings_batch.append(features.cpu())
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

            if not embeddings_batch:
                continue

            image_embeddings = torch.cat(embeddings_batch).to(self.device)
            similarity = (image_embeddings @ self.text_features.T).cpu().numpy()
            predicted_classes = np.argmax(similarity, axis=1)

            # Move files based on prediction
            for idx, img_path in enumerate(batch_paths):
                predicted_class_idx = predicted_classes[idx]
                predicted_label = self.class_names[predicted_class_idx]

                # Map CLIP labels back to folder names
                folder_name = None
                for key, value in self.class_mapping.items():
                    if value == predicted_label:
                        folder_name = key
                        break

                if folder_name is None:
                    folder_name = "Unknown"

                filename = os.path.basename(img_path)
                dest_path = os.path.join(output_folder, folder_name, filename)

                shutil.copy(img_path, dest_path)

        print(f"Files saved to: {output_folder}")