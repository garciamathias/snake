import torch
import torch.nn as nn
from collections import OrderedDict

def load_model(path):
    # Charger le dictionnaire d'état du modèle
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    return state_dict

def analyze_architecture(state_dict):
    architecture = OrderedDict()
    for key, value in state_dict.items():
        layer_name = key.split('.')[0]
        if layer_name not in architecture:
            architecture[layer_name] = {}
        
        if 'weight' in key:
            architecture[layer_name]['shape'] = value.shape
            architecture[layer_name]['params'] = value.numel()
        elif 'bias' in key:
            architecture[layer_name]['bias'] = True

    return architecture

def print_architecture(architecture):
    total_params = 0
    print("Architecture du modèle :")
    print("========================")
    
    for layer_name, layer_info in architecture.items():
        print(f"\nCouche : {layer_name}")
        print(f"  Shape : {layer_info['shape']}")
        print(f"  Paramètres : {layer_info['params']:,}")
        print(f"  Biais : {'Oui' if layer_info.get('bias', False) else 'Non'}")
        
        if layer_name.startswith('linear'):
            input_features = layer_info['shape'][1]
            output_features = layer_info['shape'][0]
            print(f"  Entrées : {input_features}")
            print(f"  Sorties : {output_features}")
        
        total_params += layer_info['params']
    
    print("\nRésumé :")
    print(f"Nombre total de paramètres : {total_params:,}")

def main():
    model_path = 'best_agent.pth'  # Assurez-vous que ce chemin est correct
    
    try:
        state_dict = load_model(model_path)
        architecture = analyze_architecture(state_dict)
        print_architecture(architecture)
    except Exception as e:
        print(f"Erreur lors de l'analyse du modèle : {e}")

if __name__ == "__main__":
    main()