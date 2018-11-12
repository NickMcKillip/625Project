import data
from trainer import Trainer
from args import args
from PIL import Image
from os import listdir
from os.path import isfile, join
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import imageio

DIRECTORY = "./predictions"


def print_grid(files, epoch, mAP):
    # files: shape = (10,10), each outer list should be on its own line
    #  path = "/Users/josiahcoad/Desktop/Coding/githublibs/personOld/market/query/"
    path = ""
#      files = [ f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == "jpg"]
    new_im = Image.new('RGB', (3000, 3100))
    for i in range(min(9, len(files))):
        for j in range(min(10, len(files[0]))):
            im = Image.open(join(path, files[i][j]))
            im = im.resize((140, 280), Image.ANTIALIAS)
            # im.thumbnail((500, 500))
            new_im.paste(im, ((j)*300 + 50, (i+1)*300 + 50))
    draw = ImageDraw.Draw(new_im)
    font = ImageFont.truetype("Aaargh.ttf", 160)
    draw.text((new_im.size[0] / 2 - 600, 80),
              f"Epoch: {epoch}, mAP: {round(mAP, 3)}", (255, 255, 255), font=font)
    draw.line((280, 0, 280, new_im.size[1]), fill=128, width=10)
    new_im.save(f"{DIRECTORY}/predictions_{epoch}.png")


def make_gif():
    images = []
    for filepath in listdir(DIRECTORY):
        if filepath.endswith(".png"):
            images.append(imageio.imread(join(DIRECTORY, filepath)))
    imageio.mimsave(f'{DIRECTORY}/training.gif', images)


def build_image(filepath):
    epoch = int(filepath.split("epoch")[1].split(".")[0])
    with open(filepath) as file_:
        lines = file_.readlines()

    def parse_line(line):
        query, test = line.split(':')
        query_idx, query_id, avg_precision = query.split(',')
        test_predictions = test.split('-')
        predictions = [tuple(map(int, p.split(','))) for p in test_predictions]
        return int(query_idx), int(query_id), float(avg_precision), predictions

    loader = data.Data(args)
    allfiles = []
    query_predictions, mAP = lines[:-1], lines[-1]
    for query in query_predictions:
        query_idx, query_id, avg_precision, predictions = parse_line(query)
        query_path = loader.queryset.imgs[query_idx]
        prediction_paths = [loader.testset.imgs[idx] for idx, _ in predictions]
        query_files = [query_path] + prediction_paths
        allfiles.append(query_files)

    print_grid(allfiles, epoch, mAP)


if __name__ == "__main__":
    for filename in listdir(DIRECTORY):
        if filename.endswith(".txt"):
            build_image(join(DIRECTORY, filename))
    make_gif()