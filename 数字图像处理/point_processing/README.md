# Point Processing

Usage:

```bash
python main.py --brightness 30 --source fig/eiffel.jpg
python main.py --contrast 2 --source fig/venice.jpg
python main.py --gamma 0.8 --source fig/lake.jpg
python main.py --hist-eq --source fig/sydney.jpg
python main.py --hist-match --source fig/new_york.jpg --target fig/scenery.jpg
python main.py --saturation 30 --source fig/road.jpg
```

Each command generates an output image `out.jpg` by default. Specify `--save-path` to change the output path.
