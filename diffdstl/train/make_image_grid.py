import os
import io 
import argparse
from PIL import Image 


def make_image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols, (len(imgs), rows, cols)

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', type=str, )
    parser.add_argument('save_path', type=str, )
    parser.add_argument('--cols', type=int, default=1)
    args = parser.parse_args()
    print('args', args)
    files = os.listdir(args.src_dir)
    images = []
    for f in sorted(files):
        f = os.path.join(args.src_dir, f)
        images.append(Image.open(f).convert('RGB'))
    rows = len(images) // args.cols
    grid = make_image_grid(images, rows=rows, cols=args.cols)
    print('save grid to', args.save_path)
    grid.save(args.save_path)
