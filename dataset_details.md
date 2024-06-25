# General dataset info

### Academic and download details

- Dataset : DongNiao International Birds 10000 (DIB-10K)
- Article acedémique description du dataset : https://arxiv.org/pdf/2010.06454
- Téléchargement dataset : http://ca.dongniao.net/download.html
-    --> Seuls les liens "part" fonctionnent. ~1h par partie et 9 téléchargements simultanés maximums

### Dataset stats

- Total de 3,411,804 d'images
- 10,915 espèces d'oiseaux
- dataset imbalanced entre espèces

### Organisation du dataset

- 70 fichiers 'DIB-10K_XXX.tgz' (~3.5-4.0 Go chacun)
- Organisation en dossier:
    ```species_number.species_name/img_name.jpg```

Exemple de fichier:
``` 863.Dimorphic Egret/
863.Dimorphic Egret/863.Dimorphic_Egret_0_315_122_635_442_-_g_0130.jpg
863.Dimorphic Egret/863.Dimorphic_Egret_0_303_109_629_435_-_f_0114.jpg
863.Dimorphic Egret/863.Dimorphic_Egret_0_412_202_970_760_-_g_0042.jpg
```

### Fichier de description du dataset

- fichier csv global avec pour chaque espèce sa localisation et le nombre d'images 
- Des espaces remplacent les "," pour plus de lisibilité ici

```
bird_name	bird_name_raw	dataset_part	folder	nb_images
common_ostrich	Common Ostrich	DIB-10K_1	0.Common Ostrich	2361
somali_ostrich	Somali Ostrich	DIB-10K_1	1.Somali Ostrich	395
greater_rhea	Greater Rhea	DIB-10K_1	2.Greater Rhea	2401
lesser_rhea	Lesser Rhea	DIB-10K_1	3.Lesser Rhea	962
```

- bird_name est en lower_snake_case (les ' ont été supprimés)


### Images

- Images en format 300x300 pixels, RGB
- 1 oiseau par image et centré
- Image scaling : image pad jusqu'à un carré (avec des 0?) et redimmensionnées
    --> exemple d'image dans publication scientifique et dans la page de téléchargement
