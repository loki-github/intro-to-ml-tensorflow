import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import tensorflow_hub as hub

def process_image(image):
    image_size = 224
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = tf.image.resize(image,(image_size, image_size))
    image_tensor = image_tensor / 255
    return image_tensor.numpy()


def predict_image_class(image_path, model, top_k):
    # image_path = './test_images/orange_dahlia.jpg'
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = np.expand_dims(process_image(image), axis=0)
    ps = model.predict(processed_image)
    ps = ps[0].tolist()

    values, indices = tf.math.top_k(ps, k=top_k)

    probs = values.numpy().tolist()
    classes = indices.numpy().tolist()
    return probs, classes

def get_correct_class_names_map(class_names):
    # labels have values from 0-101 but the label_map.json contains index 1-102
    class_names_2 = dict()
    for key in class_names:
        class_names_2[str(int(key) - 1)] = class_names[key]
    return class_names_2

def predict(image_path,saved_model,top_k,class_names):
    model = tf.keras.models.load_model(saved_model,custom_objects={'KerasLayer':hub.KerasLayer})
    correct_class_map = get_correct_class_names_map(class_names)
    probs, classes = predict_image_class(image_path, model, top_k)

    pred_label_names = []
    for lbl in classes:
        pred_label_names.append(correct_class_map[str(lbl)])

    print("top K probabilities: ",probs)
    print("top K classes: ", classes)
    print("top K class names: ", pred_label_names)
    class_prob_dict = dict(zip(pred_label_names, probs))
    print("\nTop K classes along with associated probabilities :\n", class_prob_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description for parser")
    parser.add_argument("image_path", help="Image Path", default="")
    parser.add_argument("saved_model", help="Model Path", default="")
    parser.add_argument("--top_k", help="Fetch top K predictions", required=False, default=3)
    parser.add_argument("--class_names", help="Json file having integer to class mapping", required=False,default="label_map.json")

    args = parser.parse_args()

    with open(args.class_names, 'r') as f:
        class_names = json.load(f)

    predict(args.image_path, args.saved_model, args.top_k, class_names)