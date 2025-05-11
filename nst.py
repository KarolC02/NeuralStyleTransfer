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


resize_size = 1024
def image_loader(path):
    image = Image.open(path).convert("RGB")
    raw = transforms.Resize((resize_size, resize_size))(image)
    tensor = transforms.ToTensor()(raw)  # unnormalized, in [0, 1]
    return tensor.to(device)


normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

def prepare_input(img_tensor, add_noise=False):
    if add_noise:
        with torch.no_grad():
            noise = torch.empty_like(img_tensor).uniform_(-0.25, 0.25)
            img_tensor = (img_tensor + noise).clamp(0, 1)
    return normalizer(img_tensor).unsqueeze(0).to(device)

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)


def calc_content_cost(gen_feat, orig_feat):
    N, C, H, W = gen_feat.shape
    norm = 4 * C * H * W
    return torch.sum((gen_feat - orig_feat) ** 2) / norm

def calc_style_layer_cost(gen, style):
    _, C, H, W = gen.shape
    G = torch.mm(gen.view(C, -1), gen.view(C, -1).t())
    A = torch.mm(style.view(C, -1), style.view(C, -1).t())
    norm = 4 * (C * H * W) ** 2
    return torch.sum((G - A) ** 2) / norm

layer_weights = [0.3, 0.4, 0.3]

def calc_style_cost(style_feats, gen_feats):
    total = 0.0
    for w, gen, style in zip(layer_weights, gen_feats, style_feats):
        total += calc_style_layer_cost(gen, style) * w
    return total 

def get_layer_outputs(x, model, layer_indices):
    outputs = []
    for i, layer in enumerate(model.features):
        x = layer(x)
        if str(i) in layer_indices or i in layer_indices:
            outputs.append(x)
    return outputs

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
    
    raw_content = image_loader(args.content)
    raw_style = image_loader(args.style)

    content_img = prepare_input(raw_content)
    style_img = prepare_input(raw_style)

    if args.init == 'content':
        generated_image = prepare_input(raw_content, add_noise=True).clone().requires_grad_(True)
    elif args.init == 'style':
        generated_image = prepare_input(raw_style).clone().requires_grad_(True)
    else:  # noise
        generated_image = torch.randn_like(content_img, requires_grad=True)


    
    style_layers = ['1', '6', '11', '20', '29'] 
    content_layer = '34'

    model = models.vgg19(weights=VGG19_Weights.DEFAULT).to(device).eval()

    with torch.no_grad():
        style_features = get_layer_outputs(style_img, model, style_layers)
        content_features = get_layer_outputs(content_img, model, [content_layer])[0]

    assert isinstance(style_features, list), "Style features should be a list."
    assert len(style_features) == len(style_layers), f"Expected {len(style_layers)} style layers."
    assert all(isinstance(f, torch.Tensor) for f in style_features), "Each style feature must be a tensor."
    assert isinstance(content_features, torch.Tensor), "Content feature should be a single tensor."
    assert content_features.dim() == 4, "Content feature should have shape (1, C, H, W)."

    optimizer = optim.Adam([generated_image], lr=args.lr)

    alpha = args.alpha
    beta = args.beta

    for e in range(args.epochs):
        gen_features_style = get_layer_outputs(generated_image, model, style_layers)
        gen_features_content = get_layer_outputs(generated_image, model, [content_layer])[0]

        content_loss = calc_content_cost(gen_features_content, content_features)
        style_loss = calc_style_cost(gen_features_style, style_features)

        total_loss = alpha * content_loss + beta * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if e % args.save_every == 0:
            print(f"Epoch {e}: Loss = {total_loss.item()}")
            with torch.no_grad():
                img = inv_normalize(generated_image.squeeze(0).cpu()).clamp(0, 1)
                save_image(img, f"{args.output_dir}/gen_{e}.png")

    with torch.no_grad():
        final_img = inv_normalize(generated_image.squeeze(0).cpu()).clamp(0, 1)
        save_image(final_img, f"{args.output_dir}/gen_final.png")

if __name__ == "__main__":
    main()
