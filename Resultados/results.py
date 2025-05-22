import csv
import os

# Função para salvar métricas no CSV
def salvar_metricas_csv(metrics, arquivo_csv='metricas_ia.csv'):
    file_exists = os.path.isfile(arquivo_csv)
    
    with open(arquivo_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Escreve cabeçalho se arquivo ainda não existe
        if not file_exists:
            writer.writerow([
                'epoch', 'accuracy',
                'precision_correto', 'precision_incorreto',
                'recall_correto', 'recall_incorreto',
                'f1_correto', 'f1_incorreto',
                'support_correto', 'support_incorreto',
                'cm_00', 'cm_01', 'cm_10', 'cm_11'
            ])
        
        # Escreve os dados
        writer.writerow([
            metrics['epoch'],
            metrics['accuracy'],
            metrics['precision']['correto'],
            metrics['precision']['incorreto'],
            metrics['recall']['correto'],
            metrics['recall']['incorreto'],
            metrics['f1_score']['correto'],
            metrics['f1_score']['incorreto'],
            metrics['support']['correto'],
            metrics['support']['incorreto'],
            metrics['confusion_matrix'][0][0],
            metrics['confusion_matrix'][0][1],
            metrics['confusion_matrix'][1][0],
            metrics['confusion_matrix'][1][1]
        ])

# Exemplo de uso
metrics_exemplo = {
    "epoch": 1,
    "accuracy": 0.81,
    "precision": {"correto": 0.83, "incorreto": 0.79},
    "recall": {"correto": 0.86, "incorreto": 0.74},
    "f1_score": {"correto": 0.84, "incorreto": 0.77},
    "support": {"correto": 354, "incorreto": 246},
    "confusion_matrix": [[305, 49], [63, 183]]
}

salvar_metricas_csv(metrics_exemplo)
