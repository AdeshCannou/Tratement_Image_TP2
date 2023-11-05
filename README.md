# Compte rendu: Tratement_Image_TP2

Le nombre de cercle à afficher est à modifier pour chaque image.

Les réponses aux question se trouvent à l'interieur des fichiers sources.

Les réponses aux question 3.2.1 et 3.2.2 sont implémentés dans le même fichier 3-2.py

Des images intermédiares sont affichées. Les faire passer pour afficher le rendu final (image avec contours)

## Problèmes rencontrés

Le plus petit cercle de four.png n'est pas détecter.

Le rendu final dépends fortement des paramêtres en entrées (discrétisation, seul, sigma gauss...).

## Solutions

Aucune solution n'a été trouvée pour le cercle non détecté et on affiche les 3 plus gros. Le problèmes doit surment venir de l'accumulateur car on arrive à le détecter en passant par la méthode des gradiants.

Il a fallu tester le programme à taton avec différentes valeurs jusqu'à avoir un rendu cohérent
