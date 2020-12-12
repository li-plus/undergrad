# Face Morphing

## Getting Started

Preprocess the images and landmarks.

```sh
python preprocess.py
```

Visualize key points and Delaunay triangulation.

```sh
python visualize.py
```

Generate morphing sequences.

```sh
python main.py
```

## Results

| Source                   | 33%                        | 67%                        | Target                   |
| ------------------------ | -------------------------- | -------------------------- | ------------------------ |
| ![](inputs/1/source.jpg) | ![](outputs/1/stage_3.jpg) | ![](outputs/1/stage_6.jpg) | ![](inputs/1/target.jpg) |
| ![](inputs/2/source.jpg) | ![](outputs/2/stage_3.jpg) | ![](outputs/2/stage_6.jpg) | ![](inputs/2/target.jpg) |

