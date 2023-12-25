import cv2
import torch
from torchvision import models, transforms

# Verificar se a GPU está disponível e configurar o dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o modelo e movê-lo para o dispositivo (GPU, se disponível)
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

# Função para processar a imagem
def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).to(device), image  # Mover tensor para GPU

# Carregar e processar a imagem
tensor_image, original_image = process_image('image.jpg')

# Detectar poses
with torch.no_grad():
    prediction = model([tensor_image])

# Mover os resultados da GPU para a CPU para processamento posterior
prediction = [{k: v.to('cpu') for k, v in t.items()} for t in prediction]

# Assumindo que 'prediction' é uma lista de dicionários com 'keypoints' e 'keypoints_scores'
threshold = 0.5  # Defina um limiar de confiança
for person in prediction[0]['keypoints']:
    keypoints = person.detach().numpy()
    for point in keypoints:
        x, y, conf = point
        if conf > threshold:
            cv2.circle(original_image, (int(x), int(y)), 5, (0, 255, 0), thickness=-1)

# Conectar pontos-chave para formar o esqueleto (depende da ordem dos pontos no modelo)
# ...

# Exibir a imagem
cv2.imshow('Pose Detection', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
