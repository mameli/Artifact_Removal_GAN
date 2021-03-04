import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

from fastai import *
from fastai.vision import *
from flask import Flask, jsonify, request, send_file
from uuid import uuid4

torch.backends.cudnn.benchmark=True

app = Flask(__name__)

root_model_path = Path("./models/")
exported_model = Path("standard.pkl")
learner = load_learner(path=root_model_path, file=exported_model)

print("learner loaded")

results_img_directory = '/tmp/data_AR_GAN/output_imgs/'
os.makedirs(os.path.dirname(results_img_directory), exist_ok=True)
    
def toEven(sz):
    tempSz = [sz[0], sz[1]]
    if sz[0]%2 != 0:
        tempSz[0] += 1
    if sz[1]%2 != 0:
        tempSz[1] += 1
    return tempSz

def get_dummy_databunch(bs: int, sz: int):
    """Returns sz databunch
    """
    path = Path('./dataset/dummy/')
    src = ImageImageList.from_folder(path).split_none()

    data = (src.label_from_func(
        lambda x: path/(x.name.replace(".jpg", ".png"))
    ).transform(
        size=sz,
        tfm_y=True
    ).databunch(bs=bs, num_workers=1, no_check=True)
        .normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data


# define a predict function as an endpoint
@app.route("/predict", methods=["POST"])
def process_image():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            img_low = open_image(file)
            img_size = toEven(img_low.size)
            data_gen = get_dummy_databunch(1, img_size)
            learner.data = data_gen
            p,img_hr,b = learner.predict(img_low)
            file_out_name = results_img_directory + str(uuid4()) + ".jpeg"
            p.save(file_out_name)
            print(file_out_name)
            callback = send_file(file_out_name, mimetype='image/jpeg')
            return callback, 200
        
@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})


if __name__ == '__main__':
    port = 6000
    host = '0.0.0.0'

    app.run(host='0.0.0.0')
