import cv2
import torch
from torchvision import models, transforms

def load_segmentation_model():
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model

def segment_image(image, model):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image.shape[0], image.shape[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions

def draw_segmentation_map(image, segmentation_map):
    # Define a cor da segmentação
    segmentation_color = (0, 255, 0)

    # Cria uma máscara onde os segmentos são localizados
    mask = segmentation_map == 15  # 15 é geralmente a classe 'pessoa' no DeepLabV3
    segmentation_mask = mask.float().mul(255).byte().cpu().numpy()

    # Cria uma imagem colorida baseada na máscara
    segmentation_image = cv2.bitwise_and(image, image, mask=segmentation_mask)

    # Superpõe a imagem segmentada na original
    combined_image = cv2.addWeighted(image, 0.5, segmentation_image, 0.5, 0)
    return combined_image

def main(n_image):
    segmentation_model = load_segmentation_model()

    image_path = f'image{n_image}.jpg'
    # Carregar a imagem
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Segmentar a imagem
    segmentation_map = segment_image(image, segmentation_model)

    # Desenhar o mapa de segmentação
    segmented_image = draw_segmentation_map(image, segmentation_map)

    # Salvar ou exibir a imagem resultante
    cv2.imwrite(f'segmented/segmented_image_{n_image}.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    for n_image in range(1,6):
        print('Image:', n_image)
        main(n_image)
