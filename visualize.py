import data
from trainer import Trainer
from args import args
from PIL import Image
from os import listdir
from os.path import isfile, join

def print_grid(files):
     # files: shape = (10,10), each outer list should be on its own line
     path = "/Users/josiahcoad/Desktop/Coding/githublibs/personOld/market/query/"
#      files = [ f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == "jpg"]
     new_im = Image.new('RGB', (3000,3000))
     index = 0
     for i in range(0,3000,300):
        for j in range(0,3000,300):
                im = Image.open(join(path, files[i][j]))
                im.thumbnail((300,300))
                new_im.paste(im, (i,j))
                index += 1
     new_im.save("predictions.png")

with open("predictions.txt") as file_:
    query_predictions = file_.readlines()

def parse_line(line):
    query, test = line.split(';')
    query_idx, query_id = query.split(',')
    test_predictions = test.split('-')
    predictions = [p.split(',') for p in test_predictions]
    return query_idx, query_id, predictions

loader = data.Data(args)
allfiles = []
for query in query_predictions:
    query_idx, query_id, predictions = parse_line(query)
    query_path = loader.queryset.imgs[query_idx]
    prediction_paths = [loader.testset.imgs[idx] for idx, _ in predictions]
    query_files = [query_path] + prediction_paths
    allfiles.append(query_files)

print_grid(allfiles)