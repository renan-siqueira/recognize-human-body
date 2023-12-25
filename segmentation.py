import cv2
import torch
from torchvision import models, transforms
import os

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

def draw_segmentation_map(image, segmentation_map, remove_background):
    segmentation_color = (0, 255, 0)

    mask = segmentation_map == 15  # 15 é geralmente a classe 'pessoa' no DeepLabV3
    segmentation_mask = mask.float().mul(255).byte().cpu().numpy()

    if remove_background:
        segmented_image = cv2.bitwise_and(image, image, mask=segmentation_mask)
    else:
        segmentation_image = cv2.bitwise_and(image, image, mask=segmentation_mask)
        segmented_image = cv2.addWeighted(image, 0.5, segmentation_image, 0.5, 0)

    return segmented_image

def main(image_path, remove_background=False):
    segmentation_model = load_segmentation_model()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    segmentation_map = segment_image(image, segmentation_model)

    segmented_image = draw_segmentation_map(image, segmentation_map, remove_background)

    output_file = os.path.join('segmented', 'segmented_' + os.path.basename(image_path))
    cv2.imwrite(output_file, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    images_directory = './'

    for filename in os.listdir(images_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_directory, filename)
            print('Processing Image:', image_path)
            main(image_path, remove_background=True)
