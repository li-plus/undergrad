# GraphCut Textures

Make textures!

```shell script
python texture.py ../data/green.jpg
python texture.py ../data/strawberry.jpg
python texture.py ../data/keyboard.jpg
```

|             |                       Green                       |                       Strawberry                       |                       Keyboard                       |
| :---------: | :-----------------------------------------------: | :----------------------------------------------------: | :--------------------------------------------------: |
|   Source    |      <img src="data/green.jpg" width="50%"/>      |      <img src="data/strawberry.jpg" width="50%"/>      |      <img src="data/keyboard.jpg" width="50%"/>      |
| Synthesized | <img src="output/entire_green.png" width="100%"/> | <img src="output/entire_strawberry.png" width="100%"/> | <img src="output/entire_keyboard.png" width="100%"/> |

Panorama stitching!

Image alignment codes adapted from [this blog](https://towardsdatascience.com/image-panorama-stitching-with-opencv-2402bde6b46c)

```shell script
python align.py
python stitch.py
```

| 00                          | 01                          | 02                          | 03                          | 04                          | 05                          |
| --------------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- |
| ![](data/goldengate/00.jpg) | ![](data/goldengate/01.jpg) | ![](data/goldengate/02.jpg) | ![](data/goldengate/03.jpg) | ![](data/goldengate/04.jpg) | ![](data/goldengate/05.jpg) |

![](output/golden_poisson.jpg)
