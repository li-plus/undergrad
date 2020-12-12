# Seam Carving

Narrow

```sh
python main.py --src fig/castle.jpg --dst output/castle.jpg --delta-col -200
python main.py --src fig/ratatouille.jpg --dst output/ratatouille.jpg --delta-col -200
```

Widen

```sh
python main.py --src fig/fuji.png --dst output/fuji.jpg --delta-col 243
python main.py --src fig/dolphin.png --dst output/dolphin.jpg --delta-col 119
```

Remove an object

```sh
# object removal
python main.py --src fig/beach.png --mask fig/beach_pigeon_mask.png --dst output/beach_pigeon.jpg
python main.py --src fig/beach.png --mask fig/beach_girl_mask.png --dst output/beach_girl.jpg
# resize to original size
python main.py --src output/beach_pigeon.jpg --dst output/beach_pigeon_resized.jpg --delta-col 64
python main.py --src output/beach_girl.jpg --dst output/beach_girl_resized.jpg --delta-col 40
```
