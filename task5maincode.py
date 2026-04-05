import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. Helper Functions
# ==========================================
def load_image(img_path, max_size=400, shape=None):
    """Loads an image and converts it to a PyTorch tensor."""
    image = Image.open(img_path).convert('RGB')
    
    # Resize image if it's too large to save memory
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    # VGG-19 expects images to be normalized with these specific ImageNet stats
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # Add a batch dimension: (1, C, H, W)
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image

def im_convert(tensor):
    """Converts a PyTorch tensor back to an image format for saving/display."""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    
    # Un-normalize
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

def get_features(image, model, layers=None):
    """Run an image forward through a model and get the features for a set of layers."""
    # These are the standard layers Gatys et al. used for style and content
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # This is the standard content representation layer
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor):
    """Calculate the Gram Matrix of a given tensor. 
    This calculates the correlation of features, which represents 'style'."""
    # get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()
    
    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)
    
    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())
    
    return gram 

# ==========================================
# 2. Main Execution
# ==========================================
import numpy as np # Needed for un-normalizing the image later

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load VGG-19 features (we don't need the classification head)
    vgg = models.vgg19(pretrained=True).features
    
    # Freeze all VGG parameters since we're only optimizing the target image
    for param in vgg.parameters():
        param.requires_grad_(False)
        
    vgg.to(device)

    # Load the content and style images
    # We force the style image to be the same shape as the content image
    content = load_image('content.jpg').to(device)
    style = load_image('style.jpg', shape=content.shape[-2:]).to(device)

    # Get content and style features
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # Calculate the gram matrices for each layer of our style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Initialize the target image. Starting with a clone of the content image 
    # usually leads to faster convergence than starting with pure noise.
    target = content.clone().requires_grad_(True).to(device)

    # Weights for each style layer 
    # (weighting earlier layers more gives a larger-scale style effect)
    style_weights = {'conv1_1': 1.,
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}

    # Weights for the total loss
    content_weight = 1  # alpha
    style_weight = 1e6  # beta (style usually needs to be weighted much heavier)

    # Use Adam optimizer to update the target image (not the model weights!)
    optimizer = optim.Adam([target], lr=0.003)
    steps = 2000  # Number of optimization steps

    print("Starting optimization...")
    for ii in range(1, steps+1):
        # 1. Get features of the target image
        target_features = get_features(target, vgg)
        
        # 2. Calculate Content Loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        
        # 3. Calculate Style Loss
        style_loss = 0
        for layer in style_weights:
            # get the target image's style for the current layer
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            
            # get the "target" style
            style_gram = style_grams[layer]
            
            # the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            
            # add to the total style loss
            style_loss += layer_style_loss / (d * h * w)
            
        # 4. Calculate Total Loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # 5. Backpropagate and Update target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Print progress
        if  ii % 400 == 0:
            print(f"Step {ii} - Total loss: {total_loss.item():.4f}")

    # Save the final synthesized image
    final_img = im_convert(target)
    plt.imsave("stylized_output.jpg", final_img)
    print("Process complete! Saved as 'stylized_output.jpg'")

if __name__ == "__main__":
    main()
