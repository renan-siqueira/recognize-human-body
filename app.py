import cv2
import torch
from torchvision import models, transforms


def load_model(device):
    model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()
    return model

def process_image(image_path, device):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).to(device), image

def detect_poses(model, tensor_image):
    with torch.no_grad():
        prediction = model([tensor_image])
    return prediction

def draw_keypoints(prediction, original_image, threshold=0.5):
    for person in prediction[0]['keypoints']:
        keypoints = person.detach().numpy()
        for point in keypoints:
            x, y, conf = point
            if conf > threshold:
                cv2.circle(original_image, (int(x), int(y)), 5, (0, 255, 0), thickness=-1)

def draw_skeleton(prediction, original_image, should_draw_skeleton, threshold=0.5):
    if not should_draw_skeleton:
        return

    arms = [(5, 7), (7, 9), (6, 8), (8, 10)]
    body_legs = [(5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

    for person in prediction[0]['keypoints']:
        keypoints = person.detach().numpy()
        for start_point, end_point in arms + body_legs:
            if keypoints[start_point][2] > threshold and keypoints[end_point][2] > threshold:
                start_pos = tuple(keypoints[start_point][:2].astype(int))
                end_pos = tuple(keypoints[end_point][:2].astype(int))
                cv2.line(original_image, start_pos, end_pos, (255, 0, 0), 3)

def save_image(image, path):
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def main(n_image, should_draw_skeleton=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    image_name = f'image{n_image}.jpg'
    tensor_image, original_image = process_image(image_name, device)
    prediction = detect_poses(model, tensor_image)

    prediction = [{k: v.to('cpu') for k, v in t.items()} for t in prediction]

    draw_keypoints(prediction, original_image)
    draw_skeleton(prediction, original_image, should_draw_skeleton)

    detected_image = f'detected/pose_detected_{n_image}.jpg'
    save_image(original_image, detected_image)

if __name__ == "__main__":
    for n_image in range(1, 6):
        print('Image:', n_image)
        main(n_image, should_draw_skeleton=False)
