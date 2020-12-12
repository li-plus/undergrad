import requests
import base64
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--api-config', type=str, default='api_config.json')
    args = parser.parse_args()

    # load image
    with open(args.image, 'rb') as f:
        image = f.read()

    image_base64 = base64.b64encode(image)

    # load face-plus-plus api config
    with open(args.api_config) as f:
        api_config = json.load(f)

    # post data
    url = 'https://api-us.faceplusplus.com/facepp/v3/detect'
    data = {
        'api_key': api_config['api_key'],
        'api_secret': api_config['api_secret'],
        'image_base64': image_base64,
        'return_landmark': 1
    }
    response = requests.post(url, data)
    data = response.json()

    with open(args.save_path, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))


def convert(save_path):
    with open(save_path, 'r') as f:
        data = json.load(f)

    assert len(data['faces']) == 1

    with open(save_path + '.txt', 'w') as f:
        for point in data['faces'][0]['landmark'].values():
            x, y = point['x'], point['y']
            f.write('{} {}\n'.format(x, y))


if __name__ == "__main__":
    main()
