# AI_Image_Upscaler
Projekt realizowany jako praca inżynierska pt. "Implementacja algorytmów sztucznej inteligencji w celu zwiększania rozdzielczości obrazów" na kierunku informatyka algorytmiczna na Politechnice Wrocławskiej. Program umożliwia trenowanie oraz wykorzystywanie modelu sieci konwolucyjnej (SRCNN) oraz generatywnej sieci współzawodniczącej (SRGAN) do dwukrotnego zwiększania rozdzielczości zdjęć. Wykorzystane architektury modeli zaproponowane zostały w pracach naukowych:
- Image Super-Resolution Using Deep Convolutional Networks (https://arxiv.org/abs/1501.00092)
- Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (https://arxiv.org/abs/1609.04802)

### Przygotowanie danych
Proces przygotowania zbioru treningowego oraz walidacyjnego. 
```
python3 prepare_data.py
```
### Trening
Proces trenowania wybranego modelu dla określonej liczby kanałów wejściowych.
```
python3 train.py --arch $[srcnn/srgan] --channels $[1/3]
```
### Powiększanie zdjęć
Proces dwukrotnego powiększenia danego zdjęcia, przy wykorzystaniu wybranego modelu.
```
python3 run.py --arch $[srcnn/srgan] --channels $[1/3] --img-path $image_to_upscale --weights-path $path_to_saved_weights
```
W przypadku pominięcia parametru 'weights-path' wagi dobrane zostaną odpowiednio z katalogu 'data/Saved'.