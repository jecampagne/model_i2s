#!/bin/bash
#SBATCH --job-name=gener    # nom du job
#SBATCH -A ixh@v100
#SBATCH --partition=prepost
#SBATCH --nodes=1                   # nombre de noeuds
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread        # réservation des cœurs physiques (pas d'hyperthreading)

#SBATCH --time=1:30:00              # temps d’execution maximum demande (HH:MM:SS)
#SBATCH --output=gener_%j.out # nom du fichier de sortie
#SBATCH --error=gener_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)
 
# Nettoyage des modules charges en interactif et herites par defaut
module purge
 
 
# Chargement des modules
 module load  pytorch-gpu/py3/2.6.0
 
# Echo des commandes lancees
set -x
 
srun python -u mock_generator.py
