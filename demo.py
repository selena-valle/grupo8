import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn
import os

# Leer etiquetas desde un archivo de texto
def load_labels_from_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no se encuentra.")
    
    labels = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                img_path, label = parts
                labels[img_path] = label
    return labels

# Configuración
model_path = './data/crnn.pth'
image_paths = [
    './data/demo.png',
    './data/demo1.jpg',
    './data/demo2.jpg',
    './data/demo3.jpg',
    './data/demo4.jpg',
    './data/demo5.jpg'
]
labels_path = 'labels.txt'
true_labels = load_labels_from_file(labels_path)
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz¡!'

# Cargar el modelo
model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
#model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load(model_path, weights_only=True))

converter = utils.strLabelConverter(alphabet)
transformer = dataset.resizeNormalize((100, 32))

def process_image(image_path):
    # Cargar y transformar la imagen
    image = Image.open(image_path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    # Realizar predicción
    model.eval()
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))
    return sim_pred

# Calcular y mostrar la precisión
correct = 0
total = len(image_paths)

for img_path in image_paths:
    if os.path.isfile(img_path):
        print(f"Procesando {img_path}")
        prediction = process_image(img_path)
        true_label = true_labels.get(img_path, "")
        print(f"Predicción: {prediction}")
        print(f"Etiqueta Real: {true_label}\n")
        if prediction == true_label:
            correct += 1
    else:
        print(f"File not found: {img_path}")

accuracy = (correct / total) * 100 if total > 0 else 0
print(f"Precisión: {accuracy:.2f}%")
