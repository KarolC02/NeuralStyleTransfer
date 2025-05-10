import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg19, VGG19_Weights
import torch.optim as optim
from torchvision.utils import save_image
import os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(path):
    image = Image.open(path).convert("RGB")
    loader = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.req_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(weights=VGG19_Weights.DEFAULT).features[:29]
    
    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.req_features:
                features.append(x)
        return features

def calc_content_loss(gen_feat, orig_feat):
    N, C, H, W = gen_feat.shape
    norm = C * H * W
    return torch.sum((gen_feat - orig_feat) ** 2) / norm

def calc_style_loss(gen, style):
    _, C, H, W = gen.shape
    G = torch.mm(gen.view(C, -1), gen.view(C, -1).t())
    A = torch.mm(style.view(C, -1), style.view(C, -1).t())
    norm = 4 * (C ** 2) * (H * W) ** 2
    return torch.sum((G - A) ** 2) / norm

def calculate_loss(gen_features, orig_features, style_features, alpha, beta):
    style_loss, content_loss = 0, 0
    for gen, cont, style in zip(gen_features, orig_features, style_features):
        content_loss += calc_content_loss(gen, cont)
        style_loss += calc_style_loss(gen, style)
    return alpha * content_loss + beta * style_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', required=True, help='Path to content image')
    parser.add_argument('--style', required=True, help='Path to style image')
    parser.add_argument('--output-dir', required=True, help='Directory to save outputs')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=10000)
    parser.add_argument('--lr', type=float, default=0.004)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--init', choices=['content', 'style', 'noise'], default='content',
                        help='Initialization method: content, style, or noise')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    content_img = image_loader(args.content)
    style_img = image_loader(args.style)

    if args.init == 'content':
        generated_image = content_img.clone().requires_grad_(True)
    elif args.init == 'style':
        generated_image = style_img.clone().requires_grad_(True)
    else:  # noise
        generated_image = torch.randn_like(content_img, requires_grad=True)

    model = VGG().to(device).eval()

    with torch.no_grad():
        style_features = model(style_img)
        content_features = model(content_img)

    optimizer = torch.optim.LBFGS([generated_image])

    run = [0]  # mutable counter for closure
    max_iter = args.epochs

    while run[0] <= max_iter:
        def closure():
            optimizer.zero_grad()
            gen_features = model(generated_image)
            total_loss = calculate_loss(gen_features, content_features, style_features, args.alpha, args.beta)
            total_loss.backward()
            if run[0] % args.save_every == 0:
                print(f"Epoch {run[0]}: Loss = {total_loss.item()}")
                with torch.no_grad():
                    img = inv_normalize(generated_image.squeeze(0).cpu()).clamp(0, 1)
                    save_image(img, f"{args.output_dir}/gen_{run[0]}.png")
            run[0] += 1
            return total_loss

        optimizer.step(closure)

    with torch.no_grad():
        final_img = inv_normalize(generated_image.squeeze(0).cpu()).clamp(0, 1)
        save_image(final_img, f"{args.output_dir}/gen_final.png")

if __name__ == "__main__":
    main()
