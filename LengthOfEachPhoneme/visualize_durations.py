import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pickle
import chromadb
from extract_text.train import Speech2TextModel, SpeechDataset
from transformers import T5Tokenizer
from sklearn.metrics import confusion_matrix

def load_model_and_data(model_path, dataset_path, db_path, collection_name):
    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Загружаем модель с правильным количеством фонем (69)
    model = Speech2TextModel(t5_model_name="t5-base", num_phonemes=69)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Загружаем датасет
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    dataset = SpeechDataset(dataset_path, db_path, collection_name, tokenizer, train=False)
    
    return model, dataset

def visualize_duration_distribution(dataset, save_path):
    """Визуализация распределения длительностей фонем"""
    all_durations = []
    duration_labels = []
    
    for item in dataset:
        # Получаем длительности из датасета
        durations = item['durations'].numpy()  # Используем предварительно вычисленные длительности
        labels = item['duration_labels']
        
        all_durations.extend(durations)
        duration_labels.extend(labels)
    
    # Создаем график
    plt.figure(figsize=(12, 6))
    
    # Распределение длительностей
    plt.subplot(1, 2, 1)
    sns.histplot(data=all_durations, bins=50)
    plt.title('Distribution of Phoneme Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    
    # Распределение меток длительности
    plt.subplot(1, 2, 2)
    sns.countplot(x=duration_labels)
    plt.title('Distribution of Duration Labels')
    plt.xlabel('Duration Label')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path / 'duration_distribution.png')
    plt.close()

def visualize_model_predictions(model, dataset, save_path):
    """Визуализация предсказаний модели"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for item in dataset:
            speech_emb = item['speech_embeddings'].to(device).unsqueeze(0)
            labels = item['labels']
            durations = item['durations'].to(device).unsqueeze(0)
            phoneme_ids = item['phoneme_ids'].to(device).unsqueeze(0)
            
            # Получаем предсказания
            outputs = model(
                speech_embeddings=speech_emb,
                labels=labels.unsqueeze(0),
                phoneme_ids=phoneme_ids,
                durations=durations
            )
            predictions = torch.argmax(outputs["logits"], dim=-1)
            
            # Сохраняем предсказания и метки
            all_predictions.extend(predictions[0].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Создаем матрицу ошибок
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(all_labels, all_predictions), 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=['short', 'medium', 'long'],
                yticklabels=['short', 'medium', 'long'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path / 'confusion_matrix.png')
    plt.close()

def main():
    # Пути к файлам
    model_path = "speech2text_en_model/checkpoint-600/pytorch_model.bin"
    dataset_path = "dataset/phoneme_dataset.pkl"
    db_path = "embeddings/"
    collection_name = "gender_embeddings"
    save_path = Path("visualization_results")
    save_path.mkdir(exist_ok=True)
    
    # Загружаем модель и данные
    model, dataset = load_model_and_data(model_path, dataset_path, db_path, collection_name)
    
    # Создаем визуализации
    visualize_duration_distribution(dataset, save_path)
    visualize_model_predictions(model, dataset, save_path)
    
    print(f"Визуализации сохранены в директории: {save_path}")

if __name__ == "__main__":
    main() 