import cv2
import torch
from torchvision import models, transforms

def load_model(device):
    """ Carregar o modelo e movê-lo para o dispositivo (GPU, se disponível) """
    model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()
    return model

def process_image(image_path, device):
    """ Função para processar a imagem """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).to(device), image  # Mover tensor para GPU

def detect_poses(model, tensor_image):
    """ Detectar poses na imagem """
    with torch.no_grad():
        prediction = model([tensor_image])
    return prediction

def draw_keypoints(prediction, original_image, threshold=0.5):
    """ Desenhar os keypoints na imagem """
    for person in prediction[0]['keypoints']:
        keypoints = person.detach().numpy()
        for point in keypoints:
            x, y, conf = point
            if conf > threshold:
                cv2.circle(original_image, (int(x), int(y)), 5, (0, 255, 0), thickness=-1)

def save_image(image, path):
    """ Salvar a imagem processada """
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def main(n_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    image_name = f'image{n_image}.jpg'
    tensor_image, original_image = process_image(image_name, device)
    prediction = detect_poses(model, tensor_image)

    # Mover os resultados da GPU para a CPU para processamento posterior
    prediction = [{k: v.to('cpu') for k, v in t.items()} for t in prediction]

    draw_keypoints(prediction, original_image)

    # Salvar a imagem com as detecções
    detected_image = f'detected/pose_detected_{n_image}.jpg'
    save_image(original_image, detected_image)


if __name__ == "__main__":
    n_image = 5
    main(n_image)
